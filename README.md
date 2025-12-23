# WLPR Forecasting Pipeline

Physics-informed прогноз дебита жидкости (WLPR) с учетом влияния нагнетательных скважин.

## Возможности
- TSMixerx + physics-informed loss (AdaptivePhysicsLoss)
- Инжекционные признаки: kernel selection, лаги, CRM-фильтр
- Walk-forward CV, расширенные метрики, опциональный MLflow
- Кэширование, валидация данных, детальные логи

## Быстрый старт

```bash
pip install -r requirements.txt
python -m src.wlpr_pipeline --data-path MODEL_22.09.25.csv --coords-path coords.txt
```

Опционально:

```bash
python -m src.wlpr_pipeline --enable-mlflow
mlflow ui
```

## Входные данные

1. CSV с разделителем `;` и датой `дд.мм.гггг`. Минимально: `DATA`, `well`, `TYPE`, `WLPR`.
2. `coords.txt` (координаты скважин):

```
WELL X Y Z
1 1234.5 5678.9 -2100.0
```

3. Опционально: матрица расстояний `well_distances.xlsx`.

## Выходные артефакты
- `artifacts/wlpr_predictions.csv`
- `artifacts/metrics.json`
- `artifacts/metadata.json`
- `artifacts/wlpr_forecasts.pdf`
- `artifacts/wlpr_full_history.pdf`
- `artifacts/wlpr_residuals.pdf`
- `artifacts/logs/`

## Конфигурация
Основные параметры — в `PipelineConfig` (`src/wlpr_pipeline.py`).

```python
config = PipelineConfig(
    horizon=6,
    loss="physics",
    physics_weight=0.1,
    inj_top_k=5,
)
```
