# WLPR Forecasting Pipeline

Прогноз дебита жидкости (WLPR) для добывающих скважин с учетом влияния нагнетательных, геометрии и физики процесса.
Доступны два бэкенда: обучаемый TSMixerx (physics‑loss) и zero‑shot Chronos‑2.

## Быстрый старт

```bash
pip install -r requirements.txt
python -m src.wlpr_pipeline --data-path MODEL_22.09.25.csv --coords-path coords.txt
```

MLflow (опционально):

```bash
python -m src.wlpr_pipeline --enable-mlflow --mlflow-uri http://localhost:5000
mlflow ui
```

## Выбор модели

TSMixerx (по умолчанию):

```bash
python -m src.wlpr_pipeline --model-type single
```

Chronos‑2 (zero‑shot):

```bash
python -m src.wlpr_pipeline --model-type chronos2 --chronos-model amazon/chronos-2
```

Тюнинг Chronos‑2 (пример):

```bash
python -m src.wlpr_pipeline --model-type chronos2 --chronos-input-len 36 --chronos-output-len 6
```

## Входные данные

CSV:
- Разделитель `;`
- Дата: колонка `DATA`, формат `дд.мм.гггг`
- Обязательные поля: `well`, `TYPE`, `WLPR`
- `TYPE`: `Prod` или `Inject` (регистр не важен)
- Дополнительные числовые колонки используются как признаки, если они перечислены в `PipelineConfig`

Координаты (`coords.txt`), пробельный формат, можно с заголовком:

```
WELL X Y Z
1 1234.5 5678.9 -2100.0
```

Матрица расстояний (опционально): `well_distances.xlsx` с названиями скважин в строках и колонках.

## Артефакты
- `artifacts/wlpr_predictions.csv`
- `artifacts/metrics.json`
- `artifacts/cv_metrics.json`
- `artifacts/metadata.json`
- `artifacts/wlpr_forecasts.pdf`
- `artifacts/wlpr_full_history.pdf`
- `artifacts/wlpr_residuals.pdf`
- `artifacts/injection_lag_summary.csv`
- `artifacts/data_quality_report.json`
- `artifacts/logs/`

## Настройки
Основные параметры — в `PipelineConfig` (`src/wlpr_pipeline.py`).

```python
config = PipelineConfig(
    horizon=6,
    model_type="chronos2",
    loss="physics",
    inj_top_k=5,
)
```

## Полезные флаги CLI
- `--output-dir` путь для артефактов
- `--disable-cache`, `--skip-validation`
- `--log-level DEBUG|INFO|WARNING|ERROR`
- `--chronos-model`, `--chronos-input-len`, `--chronos-output-len`

## Примечания
- Chronos‑2 использует past/future covariates, но игнорирует static covariates.
- Плановые `WWIR/WWIT` можно оставлять в `futr_exog`, если они известны заранее.
