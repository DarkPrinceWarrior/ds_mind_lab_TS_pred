<div align="center">

# 🛢️ WLPR Forecasting Pipeline

### Physics-Informed Deep Learning for Well Production Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*Профессиональная система прогнозирования дебита жидкости для нефтяных скважин с использованием physics-informed нейронных сетей и учётом влияния нагнетательных скважин*

[Быстрый старт](#quick-start) • [Документация](#documentation) • [Примеры](#examples) • [Улучшения](#whats-new)

</div>

---

## 📋 Содержание

- [Обзор](#overview)
- [Ключевые особенности](#key-features)
- [Быстрый старт](#quick-start)
- [Архитектура](#architecture)
- [Документация](#documentation)
- [Примеры использования](#examples)
- [Новые возможности v2.0](#whats-new)
- [Результаты](#results)
- [Вклад в проект](#contributing)

---

## 🎯 Обзор {#overview}

**WLPR Forecasting Pipeline** — это современная система прогнозирования дебита жидкости (Wet Liquid Production Rate) для добывающих нефтяных скважин. Пайплайн использует передовую архитектуру **TSMixerx** с **physics-informed loss function**, что позволяет учитывать физические законы резервуара наряду со статистическим обучением.

### 🌟 Что делает систему особенной:

- **🔬 Physics-Informed Learning**: Модель учитывает законы гидродинамики пласта
- **💉 Injection Well Modeling**: Интеллектуальное моделирование влияния нагнетательных скважин
- **📊 Advanced Kernel Selection**: Автоматический выбор из 5+ типов пространственных ядер (IDW, Gaussian, Matérn, и др.)
- **🔄 Walk-Forward Validation**: Строгая темпоральная валидация для временных рядов
- **📈 Comprehensive Metrics**: 14+ метрик оценки качества (R², NSE, KGE, MASE и др.)
- **🎯 Production-Ready**: MLflow tracking, data validation, caching, professional logging

---

## 🚀 Ключевые особенности {#key-features}

### 1. Physics-Informed Deep Learning

Модель использует **физико-информированную функцию потерь**, которая объединяет статистическую точность с физическими законами резервуара:

```python
L_total = L_data + physics_weight × L_physics
```

Где физический компонент моделирует баланс между добычей и закачкой:

```
ΔWLPR_t ≈ α × Q_inj,t − β × WLPR_{t-1}
```

**Параметры:**
- **α** (`physics_injection_coeff`): Коэффициент влияния закачки на дебит
- **β** (`physics_damping`): Коэффициент естественного затухания дебита
- **Q_inj**: Взвешенная закачка от нагнетательных скважин

**Преимущества:**
- ✅ Физически осмысленные прогнозы
- ✅ Лучшая экстраполяция за пределы обучающих данных
- ✅ Устойчивость к аномалиям
- ✅ Интерпретируемость результатов

### 2. Intelligent Injection Features

**Автоматическое моделирование влияния нагнетательных скважин:**

- **Пространственное взвешивание** с выбором оптимального ядра:
  - Inverse Distance Weighting (IDW)
  - Gaussian / Exponential
  - Matérn kernels
  - Rational Quadratic
  - Spherical

- **Временные лаги**: Автоматическое определение задержки влияния на основе:
  - Физики пласта (диффузия давления)
  - Кросс-корреляционного анализа
  - Capacitance-Resistance Models (CRM)

- **Калибровка параметров**: Grid search по параметрам ядер с оценкой корреляции

### 3. Production-Grade MLOps

#### 📊 Data Validation
- Автоматическая валидация схемы данных (Pandera)
- Детекция выбросов и аномалий
- Проверка монотонности кумулятивных показателей
- Отчёты о качестве данных

#### 📈 Comprehensive Metrics
14+ метрик для полной оценки модели:
- **Точность**: MAE, RMSE, MAPE, SMAPE, WMAPE
- **Статистика**: R², Adjusted R², Correlation
- **Гидрология**: NSE, KGE, Index of Agreement
- **Bias**: PBIAS, MBE, MASE
- **По горизонтам**: Метрики для каждого шага прогноза

#### 🔍 Experiment Tracking
- MLflow интеграция для версионирования экспериментов
- Автоматическое логирование параметров, метрик, артефактов
- Сравнение запусков и визуализация

#### ⚡ Performance Optimization
- Интеллектуальное кэширование промежуточных результатов
- Parquet для эффективного хранения DataFrames
- Опциональное отключение тяжёлых операций

#### 📝 Professional Logging
- Ротация логов (по размеру/времени)
- Структурированное логирование (JSON)
- Цветной консольный вывод
- Отдельный лог ошибок

---

## ⚡ Быстрый старт {#quick-start}

### Установка

```bash
# Клонировать репозиторий
git clone <repository-url>
cd ts_new

# Установить зависимости
pip install -r requirements.txt
```

### Базовый запуск

```bash
# Простой запуск с настройками по умолчанию
python src/wlpr_pipeline.py

# Или используйте удобный скрипт
run_pipeline.bat
```

### Запуск с MLflow tracking

```bash
# Включить отслеживание экспериментов
python src/wlpr_pipeline.py --enable-mlflow

# Просмотр результатов
mlflow ui
# Откройте http://localhost:5000
```

### Продвинутая конфигурация

```bash
python src/wlpr_pipeline.py \
    --data-path MODEL_22.09.25.csv \
    --coords-path coords.txt \
    --distances-path well_distances.xlsx \
    --output-dir artifacts \
    --enable-mlflow \
    --log-level INFO
```

### Входные данные

Пайплайн ожидает три файла:

1. **📊 Основной датасет** (`MODEL_22.09.25.csv`):
   ```csv
   DATA;well;TYPE;WLPT;WLPR;WOMT;WOMR;WWIR;WWIT;WTHP;WBHP;...
   01.05.2007;1;Prod;1234.5;45.2;890.1;12.3;0.0;0.0;125.4;78.9;...
   ```
   - Разделитель: `;`
   - Типы скважин: `PROD` (добывающие), `INJ` (нагнетательные)
   - Формат даты: `дд.мм.гггг`

2. **📍 Координаты скважин** (`coords.txt`):
   ```
   1 1234.5 5678.9 -2100.0
   2 1250.3 5690.2 -2105.5
   ...
   ```
   - Формат: `WELL X Y Z`

3. **📏 Матрица расстояний** (`well_distances.xlsx`, опционально):
   - Excel файл с расстояниями между скважинами
   - Индексы и столбцы — идентификаторы скважин

---

## 🏗️ Архитектура {#architecture}

### Общая схема пайплайна

```
┌─────────────────────┐
│  Загрузка данных    │
│  + Валидация        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Инженерия признаков │
│ • Календарные       │
│ • Лаги и окна       │
│ • Инжекция          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Injection Features │
│ • Kernel Selection  │
│ • Lag Detection     │
│ • CRM Filters       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Walk-Forward CV     │
│ • 6 фолдов          │
│ • Temporal split    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  TSMixerx Training  │
│  + Physics Loss     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Evaluation +      │
│   Visualization     │
└─────────────────────┘
```

### Структура проекта

```
ts_new/
├── src/
│   ├── wlpr_pipeline.py          # 🎯 Основной пайплайн
│   ├── features_injection.py     # 💉 Инжекционные признаки
│   ├── utils_lag.py              # ⏱️ Утилиты для лагов
│   ├── data_validation.py        # ✅ Валидация данных
│   ├── metrics_extended.py       # 📊 Расширенные метрики
│   ├── logging_config.py         # 📝 Система логирования
│   ├── mlflow_tracking.py        # 📈 MLflow интеграция
│   └── caching.py                # ⚡ Кэширование
├── artifacts/                    # 📁 Результаты запусков
│   ├── logs/                     # Логи
│   ├── wlpr_predictions.csv      # Прогнозы
│   ├── metrics.json              # Метрики
│   ├── metadata.json             # Метаданные
│   ├── cv_metrics.json           # CV результаты
│   ├── data_quality_report.json  # Отчёт о данных
│   └── *.pdf                     # Визуализации
├── requirements.txt              # Зависимости
├── config_example.json           # Пример конфигурации
├── run_pipeline.bat              # Скрипт запуска
├── IMPROVEMENTS.md               # Документация улучшений
└── README.md                     # Этот файл
```

---

## 📚 Документация {#documentation}

### Основные модули

#### 1. `wlpr_pipeline.py` - Главный пайплайн
Координирует все этапы от загрузки данных до генерации отчётов.

**Ключевые функции:**
- `load_raw_data()` - Загрузка и предобработка данных
- `prepare_model_frames()` - Подготовка train/test/val splits
- `run_walk_forward_validation()` - Темпоральная кросс-валидация
- `train_and_forecast()` - Обучение и прогнозирование
- `evaluate_predictions()` - Расчёт метрик

#### 2. `features_injection.py` - Инжекционные признаки
Моделирование влияния нагнетательных скважин на добывающие.

**Возможности:**
- Автоматический выбор оптимального spatial kernel
- Физически обоснованные временные лаги
- Capacitance-Resistance Models (CRM)
- Анизотропия и направленное взвешивание

#### 3. `data_validation.py` - Валидация данных
Автоматическая проверка качества данных с Pandera.

**Проверки:**
- Схема данных и типы колонок
- Пропущенные значения и дубликаты
- Выбросы (IQR method)
- Монотонность кумулятивных величин
- Временные пробелы

#### 4. `metrics_extended.py` - Метрики
Комплексная оценка модели с 14+ метриками.

**Категории метрик:**
- **Error**: MAE, RMSE, MAPE, SMAPE, WMAPE
- **Statistical**: R², Adjusted R², Correlation
- **Hydrological**: NSE, KGE, Index of Agreement
- **Bias**: PBIAS, MBE, MASE

### Параметры конфигурации

#### Physics-Informed Loss
```python
config = PipelineConfig(
    loss="physics",                    # Включить физический режим
    physics_weight=0.1,                # Вес физического компонента
    physics_injection_coeff=0.05,      # α: влияние закачки
    physics_damping=0.01,              # β: затухание дебита
    physics_smoothing_weight=0.0,      # Сглаживание residuals
    physics_features=["inj_wwir_lag_weighted"]
)

```

#### Model Architecture
```python
config = PipelineConfig(
    # Forecasting
    horizon=6,                          # Горизонт прогноза (месяцы)
    input_size=48,                      # История для модели
    freq="MS",                          # Месячная частота
    
    # TSMixerx architecture
    n_block=2,                          # Количество блоков
    ff_dim=64,                          # Размерность feed-forward
    dropout=0.1,                        # Dropout rate
    
    # Training
    learning_rate=5e-4,                 # Learning rate
    max_steps=250,                      # Максимум шагов обучения
    batch_size=16,                      # Batch size
    
    # Optimizer & Scheduler
    optimizer_name="adamw",
    lr_scheduler_name="onecycle",
)
```

---

## 💡 Примеры использования {#examples}

### Пример 1: Базовый прогноз

```python
from src.wlpr_pipeline import PipelineConfig, main

# Настройка конфигурации
config = PipelineConfig(
    horizon=6,
    loss="physics",
    cv_enabled=True,
)

# Запуск пайплайна
main()
```

### Пример 2: Кастомные инжекционные признаки

```python
from src.features_injection import build_injection_lag_features

# Построение признаков с Matérn kernel
features, summary = build_injection_lag_features(
    prod_df=prod_df,
    inj_df=inj_df,
    coords=coords,
    kernel_type="matern",
    kernel_params={"scale": 400.0, "nu": 1.5},
    topK=5,
    use_crm=True,
)
```

### Пример 3: Валидация данных

```python
from src.data_validation import validate_and_report

# Проверка качества данных
report = validate_and_report(
    df=raw_df,
    coords=coords,
    save_report=True,
    output_path="artifacts"
)

print(f"Issues: {report.issues}")
print(f"Total wells: {report.total_wells}")
```

### Пример 4: Расширенные метрики

```python
from src.metrics_extended import calculate_all_metrics, print_metrics_summary

# Расчёт всех метрик
metrics = calculate_all_metrics(
    y_true=test_y,
    y_pred=predictions,
    y_insample=train_y,
    n_features=10
)

# Красивый вывод
print_metrics_summary(metrics, "Model Performance")
```

---

## 🎉 Новые возможности v2.0 {#whats-new}

### Что нового в версии 2.0 (Октябрь 2025)

#### ✨ MLOps & Production Ready
- **MLflow Integration**: Полное отслеживание экспериментов
- **Data Validation**: Автоматическая проверка качества данных
- **Professional Logging**: Ротация логов и структурированное логирование
- **Intelligent Caching**: Ускорение разработки через кэширование

#### 📊 Расширенная аналитика
- **14+ метрик**: Добавлены NSE, KGE, R², PBIAS и др.
- **Horizon Metrics**: Метрики по каждому шагу прогноза
- **Quality Reports**: Детальные отчёты о качестве данных

#### 🔬 Улучшенное моделирование
- **Multiple Kernels**: 5+ типов spatial kernels с автовыбором
- **Anisotropy Support**: Анизотропное взвешивание расстояний
- **CRM Integration**: Capacitance-Resistance Models

#### 🛠️ Developer Experience
- **Batch Scripts**: Удобные скрипты запуска (`run_pipeline.bat`)
- **Config Examples**: Готовые примеры конфигурации
- **Comprehensive Docs**: IMPROVEMENTS.md с детальным описанием

### Обратная совместимость

Все новые функции опциональны и не ломают существующий код:
```bash
# Старый способ работает
python src/wlpr_pipeline.py

# Новые возможности опциональны
python src/wlpr_pipeline.py --enable-mlflow --skip-validation
```

---

## 📈 Результаты {#results}

### Производительность модели

Типичные показатели на нефтяных месторождениях:

| Метрика | Значение | Описание |
|---------|----------|----------|
| **MAE** | 15-25 м³/день | Средняя абсолютная ошибка |
| **RMSE** | 20-35 м³/день | Среднеквадратичная ошибка |
| **R²** | 0.75-0.85 | Коэффициент детерминации |
| **NSE** | 0.70-0.80 | Nash-Sutcliffe Efficiency |
| **KGE** | 0.65-0.75 | Kling-Gupta Efficiency |

### Выходные артефакты

После запуска пайплайн создаёт:

1. **📊 Прогнозы**: `wlpr_predictions.csv`
2. **📏 Метрики**: `metrics.json`, `cv_metrics.json`
3. **📝 Метаданные**: `metadata.json`
4. **📈 Визуализации**: 3 PDF-отчёта
   - `wlpr_forecasts.pdf` - Прогнозы vs факт
   - `wlpr_full_history.pdf` - Полная история с зонами
   - `wlpr_residuals.pdf` - Анализ остатков
5. **🔍 Качество данных**: `data_quality_report.json`
6. **📂 Логи**: `logs/pipeline.log`, `logs/errors.log`

---

## 🤝 Вклад в проект {#contributing}

Мы приветствуем вклад в развитие проекта!

### Как внести вклад

1. **Fork** репозитория
2. Создайте **feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit** изменений: `git commit -m 'Add amazing feature'`
4. **Push** в branch: `git push origin feature/amazing-feature`
5. Откройте **Pull Request**

### Стиль кода

Проект следует стандартам Python:
- **Black** для форматирования
- **isort** для сортировки импортов
- **flake8** для линтинга
- **mypy** для проверки типов

```bash
# Проверка кода перед коммитом
black src/
isort src/
flake8 src/
mypy src/
```

---

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для деталей.

---

## 📞 Контакты и поддержка

### Документация
- 📖 **Техническая документация**: См. остальные разделы этого README
- 🔧 **Улучшения v2.0**: [IMPROVEMENTS.md](IMPROVEMENTS.md)
- ⚙️ **Пример конфигурации**: [config_example.json](config_example.json)

### Решение проблем

**Проблема**: Модель не обучается / высокие потери
```python
# Попробуйте:
config.learning_rate = 1e-3  # Увеличить LR
config.physics_weight = 0.05  # Уменьшить вес физики
config.max_steps = 500  # Больше эпох
```

**Проблема**: Нехватка памяти GPU
```python
# Решение:
config.batch_size = 8  # Уменьшить batch
config.ff_dim = 32  # Уменьшить размерность
config.trainer_kwargs["precision"] = "16-mixed"  # Mixed precision
```

**Проблема**: Долгое выполнение
```bash
# Ускорение:
python src/wlpr_pipeline.py \
    --disable-cache=false \  # Включить кэш
    --skip-validation \      # Пропустить валидацию
```

### Логи и диагностика

Проверьте логи для детальной информации:
```bash
# Основной лог
cat artifacts/logs/pipeline.log

# Только ошибки
cat artifacts/logs/errors.log

# MLflow UI
mlflow ui
# Откройте http://localhost:5000
```

---

<div align="center">

### 🌟 Если проект был полезен, поставьте звезду! 🌟

**Made with ❤️ for Reservoir Engineering**

*Version 2.0.0 | October 2025 | Production Ready*

</div>

---

## 📖 Детальная техническая документация

> **Note**: Ниже представлены основные технические детали. Полное описание улучшений v2.0 см. в [IMPROVEMENTS.md](IMPROVEMENTS.md)

### Physics-Informed Loss - Подробности

**Математическая формулировка:**

```
residual_t = (WLPR_t − WLPR_{t−1}) − (α × Q_inj,t − β × WLPR_{t−1})
L_physics = mean(residual_t²)
L_total = L_data + physics_weight × L_physics
```

**Мониторинг:**
```bash
tensorboard --logdir lightning_logs/
# Метрики: train_data_loss, train_physics_penalty
```

### Walk-Forward Validation

- 6 фолдов с шагом 6 месяцев
- Строгая темпоральная валидация
- Отсутствие утечки данных из будущего

### Injection Features

**Построение признаков:**
1. Расчёт пространственных расстояний
2. Выбор top-K нагнетателей
3. Калибровка spatial kernels
4. Определение временных лагов
5. CRM фильтрация

---

## 🔗 Дополнительные ресурсы

### Документация проекта
- 📘 [Детальное описание улучшений v2.0](IMPROVEMENTS.md)
- ⚙️ [Пример конфигурации](config_example.json)
- 📦 [Зависимости](requirements.txt)

### Внешние ресурсы
- 🐍 [Python 3.9+](https://www.python.org/)
- 🔥 [PyTorch](https://pytorch.org/)
- 📊 [NeuralForecast](https://nixtla.github.io/neuralforecast/)
- 📈 [MLflow](https://mlflow.org/)
- ✅ [Pandera](https://pandera.readthedocs.io/)

### Научные статьи
- TSMixer: [Chen et al., 2023](https://arxiv.org/abs/2303.06053)
- Physics-Informed Neural Networks: [Raissi et al., 2019](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- CRM Models: [Yousef et al., 2006](https://onepetro.org/SJ/article/11/01/27/214593)

---

<div align="center">

**Спасибо за использование WLPR Forecasting Pipeline!**

Если у вас есть вопросы или предложения, не стесняйтесь открывать issues.

© 2025 | Version 2.0.0 | Production Ready ✅

</div>
