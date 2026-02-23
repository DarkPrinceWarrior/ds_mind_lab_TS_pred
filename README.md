# WLPR Forecasting Pipeline

Прогноз месячного дебита жидкости (`WLPR`) по добывающим скважинам с учетом влияния нагнетательных, пространственных связей и графовых признаков.

Поддерживаемые модели:
- `chronos2` (Amazon Chronos-2, zero-shot через Darts)
- `timexer` (обучаемая модель TimeXer через NeuralForecast)

Текущий CLI entrypoint:
- `python3 -m src.artifacts`

## Что делает пайплайн

- Загружает и валидирует промысловые данные
- Строит лаговые признаки закачки с подбором лага и калибровкой ядра весов
- Добавляет пространственные, временные и графовые признаки
- Делает split train/test по горизонту прогноза
- Считает walk-forward кросс-валидацию
- Строит прогноз, метрики и PDF-отчеты
- Сохраняет все артефакты в `output-dir`

## Структура проекта

- `src/artifacts.py` — основной CLI пайплайна (запуск, метрики, сохранение артефактов)
- `src/wlpr_pipeline.py` — функции загрузки данных, feature engineering, inference, evaluation
- `src/config.py` — `PipelineConfig` с параметрами моделей и признаков
- `src/features_injection.py` — лаговые признаки закачки и CRM-фильтрация
- `src/features_graph.py` — графовые признаки и агрегаты соседей
- `src/visualization.py` — PDF-отчеты по прогнозу/истории/остаткам
- `src/visualization_features.py` — PDF-анализ признаков
- `scripts/scenario_shutoff_34.py` — сценарий “остановка нагнетательной скв. 34”
- `PIPELINE_DOCUMENTATION.md` — расширенная техническая документация

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.artifacts \
  --data-path MODEL_23.09.25.csv \
  --distances-path Distance.xlsx \
  --output-dir artifacts
```

Примечание:
- При первом запуске `chronos2` модель может скачиваться из Hugging Face.

## Примеры запуска

Chronos-2:

```bash
python3 -m src.artifacts \
  --model chronos2 \
  --data-path MODEL_23.09.25.csv \
  --distances-path Distance.xlsx \
  --output-dir artifacts
```

TimeXer:

```bash
python3 -m src.artifacts \
  --model timexer \
  --data-path MODEL_23.09.25.csv \
  --distances-path Distance.xlsx \
  --output-dir artifacts_timexer
```

Chronos-2 с override параметров:

```bash
python3 -m src.artifacts \
  --model chronos2 \
  --chronos-model amazon/chronos-2 \
  --chronos-input-len 36 \
  --chronos-output-len 6
```

MLflow (опционально):

```bash
python3 -m src.artifacts --enable-mlflow --mlflow-uri http://localhost:5000
mlflow ui
```

## CLI параметры

- `--data-path` путь к CSV с промысловыми данными (по умолчанию `MODEL_23.09.25.csv`)
- `--distances-path` путь к `Distance.xlsx` (координаты + матрица расстояний)
- `--coords-path` legacy-файл координат (если не задан, берется `--distances-path`)
- `--output-dir` директория для артефактов (по умолчанию `artifacts`)
- `--model` тип модели: `chronos2` или `timexer`
- `--chronos-model` имя модели на Hugging Face для Chronos-2
- `--chronos-revision` ревизия/тег модели Chronos-2
- `--chronos-local-dir` локальная директория кэша Chronos-2
- `--chronos-input-len` входное окно Chronos-2
- `--chronos-output-len` выходное окно Chronos-2
- `--enable-mlflow` включить трекинг MLflow
- `--mlflow-uri` URI MLflow-сервера
- `--disable-cache` выключить кэш промежуточных вычислений
- `--skip-validation` пропустить валидацию данных
- `--log-level` уровень логов: `DEBUG`, `INFO`, `WARNING`, `ERROR`

## Формат входных данных

CSV (`;`-разделитель):

- `DATA` — дата в формате `ДД.ММ.ГГГГ`
- `well` — идентификатор скважины
- `TYPE` — тип скважины (`PROD`/`INJ`, регистр нормализуется)
- `WLPR` — целевая переменная (дебит жидкости)
- Дополнительные числовые поля используются как признаки при наличии в конфиге

`Distance.xlsx`:

- Должен содержать колонки `Well`, `x`, `y`, `z`
- Остальные колонки интерпретируются как матрица расстояний между скважинами
- При отсутствии файла с матрицей можно передать отдельный файл координат через `--coords-path`

Legacy координаты (`--coords-path`):

```text
well_id  x  y  z
1 1234.5 5678.9 -2100.0
```

## Выходные артефакты

Файлы в `output-dir`:

- `wlpr_predictions.csv` — прогнозы
- `metrics.json` — итоговые метрики на тесте
- `cv_metrics.json` — результаты walk-forward CV
- `metadata.json` — конфиг, список скважин, окна train/test, пути отчетов
- `injection_lag_summary.csv` — выбранные лаги/веса по парам producer-injector
- `data_quality_report.json` — отчет по качеству данных (если валидация включена)
- `wlpr_forecasts.pdf` — прогноз vs факт на тесте
- `wlpr_full_history.pdf` — полная история с разметкой train/val/test
- `wlpr_residuals.pdf` — остатки на тесте
- `feature_analysis.pdf` — анализ признаков
- `logs/` — логи запуска
- `.cache/` — кэш промежуточных вычислений (если cache не отключен)

## Сценарный анализ

Скрипт `scripts/scenario_shutoff_34.py` моделирует отключение нагнетательной скважины `34` с `2021-01-01` и сравнивает прогноз с базовым сценарием.

Пример:

```bash
python3 scripts/scenario_shutoff_34.py --model timexer
```

Результаты:

- `artifacts_timexer/scenario_shutoff_34/` для `--model timexer`
- `artifacts/scenario_shutoff_34/` для `--model chronos2`

Внутри:

- `scenario_shutoff_34.pdf`
- `comparison_report.csv`
- `detailed_comparison.csv`
- `scenario_summary.json`

## Использование как Python API

`src/wlpr_pipeline.py` содержит функции для программного вызова.

```python
from src.config import PipelineConfig
from src.wlpr_pipeline import (
    load_raw_data, load_coordinates, load_distance_matrix,
    prepare_model_frames, train_and_forecast, evaluate_predictions,
)

config = PipelineConfig(model_type="timexer")
raw = load_raw_data("MODEL_23.09.25.csv")
coords = load_coordinates("Distance.xlsx")
dist = load_distance_matrix("Distance.xlsx")
frames = prepare_model_frames(raw, coords, config, distances=dist)
preds = train_and_forecast(frames, config)
metrics, merged = evaluate_predictions(preds, frames["test_df"], frames["train_df"])
```

## Важно

- `src/wlpr_pipeline.py` сейчас не содержит отдельного CLI `main()`: для консольного запуска используйте `src.artifacts`.
- Расширенная поэтапная логика пайплайна описана в `PIPELINE_DOCUMENTATION.md`.
