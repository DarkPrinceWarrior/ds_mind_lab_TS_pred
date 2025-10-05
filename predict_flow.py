import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse
import warnings
import os
warnings.filterwarnings('ignore')

try:
    import torch
    import timesfm
except ImportError as e:
    print(f"Ошибка: {e}")
    print("Установите: pip install torch timesfm")
    exit(1)

torch.set_float32_matmul_precision("high")


def load_data(csv_path):
    """Загрузка данных из CSV"""
    print(f"Загрузка данных из {csv_path}...")
    df = pd.read_csv(csv_path, sep=';')
    
    # Переименование колонок как в оригинальном пайплайне
    df = df.rename(columns={'DATA': 'date', 'TYPE': 'type'})
    df.columns = [col.lower() for col in df.columns]
    
    # Обработка дат
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df[df['date'].notna()]
    
    # Обработка well
    df['well'] = df['well'].astype(float).astype(int).astype(str)
    
    # Обработка type
    if 'type' in df.columns:
        df['type'] = df['type'].astype(str).str.strip().str.upper()
    
    # Сортировка
    df = df.sort_values(['well', 'date'])
    df = df.drop_duplicates(['well', 'date'])
    
    print(f"Загружено {len(df)} записей, {df['well'].nunique()} скважин")
    print(f"  Производство (Prod): {(df['type']=='PROD').sum()} записей")
    print(f"  Закачка (INJ): {(df['type']=='INJ').sum()} записей")
    
    return df


def calculate_metrics(y_true, y_pred):
    """Расчет метрик качества прогноза"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Убираем нулевые значения для MAPE
    mask = y_true != 0
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # R² score
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def prepare_well_data(df, well_id, target_column='wlpr', test_months=6):
    """Подготовка данных для одной скважины с разделением на train/test"""
    well_data = df[df['well'] == str(well_id)].copy()
    well_data = well_data.sort_values('date')
    
    # ФИЛЬТРУЕМ ТОЛЬКО ПРОИЗВОДСТВЕННЫЕ ПЕРИОДЫ (TYPE == 'PROD')
    if 'type' in well_data.columns:
        prod_data = well_data[well_data['type'] == 'PROD'].copy()
    else:
        prod_data = well_data.copy()
    
    # Удаляем пропуски и нули в целевой переменной
    prod_data = prod_data.dropna(subset=[target_column])
    prod_data = prod_data[prod_data[target_column] > 0]  # Убираем нули
    
    if len(prod_data) < test_months + 10:
        return None, None, None, None
    
    # Разделяем на train и test (последние test_months для валидации)
    split_idx = len(prod_data) - test_months
    
    train_data = prod_data.iloc[:split_idx]
    test_data = prod_data.iloc[split_idx:]
    
    train_dates = train_data['date']
    train_values = train_data[target_column]
    test_dates = test_data['date']
    test_values = test_data[target_column]
    
    return train_dates, train_values, test_dates, test_values


def predict_with_timesfm(data, horizon=12):
    """Прогноз с использованием TimesFM 2.5"""
    print(f"\nИнициализация TimesFM 2.5 модели...")
    
    # Загрузка предобученной модели
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    
    print("Модель загружена успешно")
    
    # Конфигурация модели
    max_context = min(1024, len(data))  # Максимум 1024 или длина данных
    
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=min(256, horizon),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=False,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    
    print(f"Конфигурация: context={max_context}, horizon={horizon}")
    
    # Подготовка данных
    values = data.values.astype(np.float32)
    
    # Убираем нули и отрицательные значения для стабильности
    values = np.where(values <= 0, 1e-6, values)
    
    print(f"Прогнозирование на {horizon} шагов вперед...")
    
    # Делаем прогноз
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=[values],
    )
    
    predictions = point_forecast[0]
    
    return predictions


def visualize_well_results(train_dates, train_values, test_dates, test_values, 
                           predictions, metrics, well_id, target_col, save_path=None):
    """Визуализация результатов прогноза для одной скважины - ТОЛЬКО ТЕСТОВЫЙ ПЕРИОД"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    # Показываем только последние 12 точек обучающих данных (для контекста)
    context_points = min(12, len(train_values))
    train_dates_context = train_dates.iloc[-context_points:]
    train_values_context = train_values.iloc[-context_points:]
    
    # График: последние точки train + test + прогноз
    ax.plot(train_dates_context, train_values_context, 'b-', linewidth=2.5, 
            label='История (последние точки)', marker='o', markersize=7, alpha=0.8)
    ax.plot(test_dates, test_values, 'g-', linewidth=3, 
            label='Реальные значения (тест)', marker='o', markersize=9, zorder=5)
    ax.plot(test_dates, predictions, 'r--', linewidth=3, 
            label='Прогноз TimesFM', marker='s', markersize=9, zorder=5)
    
    # Разделитель train/test
    if len(train_dates) > 0:
        ax.axvline(x=train_dates.iloc[-1], color='gray', linestyle=':', 
                   linewidth=2.5, alpha=0.7, label='Начало теста', zorder=3)
    
    # Подсветка тестовой области
    ax.axvspan(test_dates.iloc[0], test_dates.iloc[-1], 
               alpha=0.1, color='yellow', zorder=1)
    
    # Метрики на графике - крупнее и ярче
    metrics_text = (
        f"MAE:  {metrics['MAE']:.2f}\n"
        f"RMSE: {metrics['RMSE']:.2f}\n"
        f"MAPE: {metrics['MAPE']:.1f}%\n"
        f"R²:   {metrics['R2']:.3f}"
    )
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
            fontsize=13, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                     edgecolor='black', alpha=0.9, linewidth=2))
    
    # Добавляем значения на точки прогноза
    for date, real, pred in zip(test_dates, test_values, predictions):
        # Реальное значение
        ax.annotate(f'{real:.1f}', xy=(date, real), 
                   xytext=(0, 10), textcoords='offset points',
                   fontsize=9, ha='center', color='green', fontweight='bold')
        # Прогноз
        ax.annotate(f'{pred:.1f}', xy=(date, pred), 
                   xytext=(0, -15), textcoords='offset points',
                   fontsize=9, ha='center', color='red', fontweight='bold')
    
    ax.set_xlabel('Дата', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{target_col.upper()} (дебит жидкости)', fontsize=13, fontweight='bold')
    ax.set_title(f'Скважина {well_id}: Валидация прогноза на 6 месяцев', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black', fancybox=True)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Улучшаем внешний вид
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def visualize_summary(all_results, save_path=None):
    """Сводная визуализация по всем скважинам"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    wells = []
    mae_scores = []
    rmse_scores = []
    mape_scores = []
    r2_scores = []
    
    for result in all_results:
        wells.append(f"Well {result['well_id']}")
        mae_scores.append(result['metrics']['MAE'])
        rmse_scores.append(result['metrics']['RMSE'])
        mape_scores.append(result['metrics']['MAPE'])
        r2_scores.append(result['metrics']['R2'])
    
    # График 1: MAE
    ax1 = axes[0, 0]
    bars1 = ax1.bar(wells, mae_scores, color='steelblue', alpha=0.7)
    ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Absolute Error по скважинам', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # График 2: RMSE
    ax2 = axes[0, 1]
    bars2 = ax2.bar(wells, rmse_scores, color='coral', alpha=0.7)
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('Root Mean Squared Error по скважинам', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # График 3: MAPE
    ax3 = axes[1, 0]
    bars3 = ax3.bar(wells, mape_scores, color='seagreen', alpha=0.7)
    ax3.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Mean Absolute Percentage Error по скважинам', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # График 4: R²
    ax4 = axes[1, 1]
    bars4 = ax4.bar(wells, r2_scores, color='mediumpurple', alpha=0.7)
    ax4.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax4.set_title('R² Score по скважинам', fontsize=13, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Сводка метрик качества прогноза TimesFM (валидация на 6 месяцах)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nСводный график сохранен: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Прогнозирование дебита жидкости с TimesFM (валидация на последних 6 месяцах)')
    parser.add_argument('--csv', type=str, default='MODEL_22.09.25.csv', 
                        help='Путь к CSV файлу с данными')
    parser.add_argument('--target', type=str, default='wlpr', 
                        help='Целевая колонка для прогноза (wlpr - liquid production rate)')
    parser.add_argument('--output_dir', type=str, default='forecast_results', 
                        help='Папка для сохранения графиков')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ПРОГНОЗИРОВАНИЕ ДЕБИТА ЖИДКОСТИ С TIMESFM 2.5")
    print("Валидация на последних 6 месяцах")
    print("="*80)
    
    # Создаем папку для результатов
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"\nСоздана папка для результатов: {args.output_dir}")
    
    # Загрузка данных
    df = load_data(args.csv)
    
    # Получаем список всех скважин
    all_wells = sorted(df['well'].unique())
    print(f"\nНайдено скважин: {len(all_wells)}")
    print(f"Список скважин: {all_wells}")
    
    # Загружаем модель один раз для всех скважин
    print("\n" + "="*80)
    print("Загрузка модели TimesFM 2.5...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    print("✓ Модель загружена успешно")
    
    # Результаты для всех скважин
    all_results = []
    
    # Прогнозирование для каждой скважины
    print("\n" + "="*80)
    print("ОБРАБОТКА СКВАЖИН")
    print("="*80)
    
    for well_id in all_wells:
        print(f"\n{'─'*80}")
        print(f"Скважина {well_id}")
        print(f"{'─'*80}")
        
        # Подготовка данных
        train_dates, train_values, test_dates, test_values = prepare_well_data(
            df, well_id, args.target, test_months=6
        )
        
        if train_dates is None:
            print(f"⚠ Пропускаем скважину {well_id}: недостаточно данных")
            continue
        
        print(f"Обучающих точек: {len(train_values)}")
        print(f"Тестовых точек: {len(test_values)}")
        print(f"Период обучения: {train_dates.iloc[0].strftime('%Y-%m-%d')} - {train_dates.iloc[-1].strftime('%Y-%m-%d')}")
        print(f"Период валидации: {test_dates.iloc[0].strftime('%Y-%m-%d')} - {test_dates.iloc[-1].strftime('%Y-%m-%d')}")
        
        try:
            # Конфигурация и компиляция модели
            max_context = min(1024, len(train_values))
            model.compile(
                timesfm.ForecastConfig(
                    max_context=max_context,
                    max_horizon=min(256, 6),
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=False,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            
            # Подготовка данных для прогноза
            train_array = train_values.values.astype(np.float32)
            train_array = np.where(train_array <= 0, 1e-6, train_array)
            
            # Прогнозирование
            print("Прогнозирование...")
            point_forecast, _ = model.forecast(
                horizon=6,
                inputs=[train_array],
            )
            
            predictions = point_forecast[0]
            
            # Расчет метрик
            metrics = calculate_metrics(test_values.values, predictions)
            
            print(f"\n📊 Метрики качества:")
            print(f"   MAE:  {metrics['MAE']:.2f}")
            print(f"   RMSE: {metrics['RMSE']:.2f}")
            print(f"   MAPE: {metrics['MAPE']:.1f}%")
            print(f"   R²:   {metrics['R2']:.3f}")
            
            # Сохраняем результат
            all_results.append({
                'well_id': well_id,
                'metrics': metrics,
                'predictions': predictions,
                'test_values': test_values.values
            })
            
            # Визуализация для скважины
            save_path = os.path.join(args.output_dir, f'well_{well_id}_forecast.png')
            visualize_well_results(
                train_dates, train_values, test_dates, test_values,
                predictions, metrics, well_id, args.target, save_path
            )
            print(f"✓ График сохранен: {save_path}")
            
        except Exception as e:
            print(f"❌ Ошибка при обработке скважины {well_id}: {e}")
            continue
    
    # Сводная визуализация
    if len(all_results) > 0:
        print("\n" + "="*80)
        print("СВОДНАЯ СТАТИСТИКА")
        print("="*80)
        
        print(f"\nУспешно обработано скважин: {len(all_results)}")
        
        # Средние метрики
        avg_mae = np.mean([r['metrics']['MAE'] for r in all_results])
        avg_rmse = np.mean([r['metrics']['RMSE'] for r in all_results])
        avg_mape = np.mean([r['metrics']['MAPE'] for r in all_results if not np.isnan(r['metrics']['MAPE'])])
        avg_r2 = np.mean([r['metrics']['R2'] for r in all_results])
        
        print(f"\n📈 Средние метрики по всем скважинам:")
        print(f"   MAE:  {avg_mae:.2f}")
        print(f"   RMSE: {avg_rmse:.2f}")
        print(f"   MAPE: {avg_mape:.1f}%")
        print(f"   R²:   {avg_r2:.3f}")
        
        # Сводный график
        summary_path = os.path.join(args.output_dir, 'summary_metrics.png')
        visualize_summary(all_results, summary_path)
        
        print("\n" + "="*80)
        print("✓ ПРОГНОЗИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print(f"✓ Результаты сохранены в папке: {args.output_dir}")
        print("="*80)
    else:
        print("\n❌ Не удалось обработать ни одной скважины")


if __name__ == "__main__":
    main()
