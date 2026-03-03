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
- Строит attention-связность `injector→producer` методом `causal_stage_geo` и признаки `inj_*_attn`
- Добавляет пространственные, временные и графовые признаки
- Делает split train/test по горизонту прогноза
- Считает walk-forward кросс-валидацию
- Калибрует prediction intervals через Conformal (ICP/WCP) на out-of-sample residuals из CV
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
python3 -m venv venv
source venv/bin/activate
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

TimeXer + Conformal (WCP, 90% PI):

```bash
export CUDA_VISIBLE_DEVICES=''  # опционально: форс CPU
python3 -m src.artifacts \
  --model timexer \
  --data-path MODEL_23.09.25.csv \
  --distances-path Distance.xlsx \
  --output-dir artifacts_timexer \
  --conformal-method wcp_exp \
  --conformal-alpha 0.1
```

TimeXer + Attention (`causal_stage_geo`, default):

```bash
export CUDA_VISIBLE_DEVICES=''  # опционально: форс CPU
python3 -m src.artifacts \
  --model timexer \
  --data-path MODEL_23.09.25.csv \
  --distances-path Distance.xlsx \
  --output-dir artifacts_timexer_attn_causal_stage_geo \
  --conformal-method wcp_exp \
  --conformal-alpha 0.1 \
  --inj-attention-smooth-strength 0.05
```

Опционально, чтобы отключить stage-adaptive gating и оставить фиксированное смешивание в `causal_stage_geo`:

```bash
python3 -m src.artifacts \
  --model timexer \
  --data-path MODEL_23.09.25.csv \
  --distances-path Distance.xlsx \
  --output-dir artifacts_timexer_attn_fixed_mix \
  --conformal-method wcp_exp \
  --conformal-alpha 0.1 \
  --disable-inj-attention-stage-adaptive
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
- `--disable-conformal` отключить conformal-интервалы
- `--conformal-alpha` miscoverage `alpha` (например `0.1` => целевое покрытие 90%)
- `--conformal-method` метод калибровки: `icp`, `wcp_exp`, `wcp_linear`
- `--conformal-exp-decay` коэффициент затухания для `wcp_exp` (ближе к `1.0` => медленнее забывание)
- `--conformal-min-samples` минимум residuals на шаг горизонта для отдельной калибровки
- `--conformal-global` использовать один глобальный `eps` для всех шагов горизонта
- `--disable-inj-attention` отключить attention-агрегацию injector→producer
- Метод attention фиксирован в конфиге: `causal_stage_geo` (переключение режимов через CLI удалено)
- `--inj-attention-target-mode` target для обучения attention: `delta` или `level`
- `--inj-attention-steps` число шагов оптимизации attention
- `--inj-attention-lr` learning rate для attention
- `--inj-attention-prior-strength` регуляризация к kernel-prior (`guidance`)
- `--inj-attention-entropy-strength` энтропийная регуляризация attention
- `--inj-attention-smooth-strength` сглаживание динамики `alpha(t)` по времени
- `--inj-attention-future-anchor-strength` якорение future `alpha(t)` к train-last
- `--inj-attention-geo-condition-strength` сила geo-conditioned blending в prior для attention
- `--disable-inj-attention-stage-adaptive` отключить stage-adaptive gating (фиксированный mix в `causal_stage_geo`)

## Conformal интервалы: что это и зачем

Conformal в этом проекте — это пост-процессинг поверх точечного прогноза `y_hat`, который строит интервалы неопределенности `cp_lo`/`cp_hi` с целевым покрытием (например 90%) на основе реальных out-of-sample ошибок.

Зачем:
- Квантильные интервалы модели (`q_0.1/q_0.9`) не гарантируют нужное покрытие на промысловых сдвигах режима.
- Conformal калибрует ширину интервала по фактическим ошибкам и обычно лучше переносится на production.
- Для нестационарности можно использовать `WCP` (`wcp_exp`/`wcp_linear`), где последние residuals важнее старых.

Как встроено в пайплайн:
- Ошибки для калибровки берутся из walk-forward CV (out-of-sample residual pool).
- По каждому шагу горизонта (`h=1..H`) считается свой `eps[h]` (или global fallback).
- На тесте интервал строится как:
  - `cp_lo = y_hat - eps[h]`
  - `cp_hi = y_hat + eps[h]`

Если `cv_enabled=False`, residual pool не формируется, и conformal-калибровка не применяется.

## Как интерпретировать Conformal в отчетах

В `wlpr_predictions.csv`:
- `cp_lo`, `cp_hi` — conformal границы интервала для точки прогноза.
- `cp_eps` — радиус интервала.
- `cp_method`, `cp_alpha` — параметры калибровки.

В `metrics.json` (раздел `reservoir`):
- `reliability_picp` — фактическая доля точек, попавших в интервал.
  - Сравнивай с целевым уровнем `1 - alpha` (например, 0.9).
  - `PICP < target`: интервалы слишком узкие (недопокрытие).
  - `PICP >> target`: интервалы слишком широкие (избыточный запас).
- `reliability_mpiw` — средняя ширина интервала (меньше — резче, но не в ущерб покрытию).
- `reliability_interval_sharpness` — нормированная ширина интервала.

Практическое правило:
- Сначала добивайся близкого к целевому `PICP`, затем оптимизируй `MPIW`.
- При частых режимных сдвигах предпочитай `wcp_exp` и подбирай `conformal_exp_decay`.

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
- `cv_metrics.json` — результаты walk-forward CV (включая `conformal_profile`, если калибровка выполнена)
- `metadata.json` — конфиг, список скважин, окна train/test, пути отчетов
- `injection_lag_summary.csv` — выбранные лаги/веса по парам producer-injector
- `alpha_dynamic.parquet`/`alpha_dynamic.csv` — временные attention-веса `(ds, prod_id, inj_id, alpha, is_train, regime_id, stage_id, attention_mode)` для метода `causal_stage_geo`
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
