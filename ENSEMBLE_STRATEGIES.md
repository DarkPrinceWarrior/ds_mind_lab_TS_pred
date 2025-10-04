# 🎯 Ensemble Strategies - Какие модели использовать

**Для EnsembleForecaster в Phase 2**

---

## 📊 Концепция Ensemble

`EnsembleForecaster` - это **гибкая обертка**, которая может комбинировать **любые модели**. Ключевой принцип: модели должны быть **разнообразными** (diversity), чтобы они ошибались по-разному.

---

## 🎯 Рекомендуемые Стратегии

### Стратегия 1: Variation of TSMixerx (САМАЯ ПРОСТАЯ) ⭐⭐⭐

**Идея:** Одна и та же архитектура, но разные гиперпараметры

**Модели:**
```python
from neuralforecast.models import TSMixerx

# Модель 1: Консервативная (меньше overfitting)
model_1 = TSMixerx(
    h=6,
    input_size=48,
    n_series=n_wells,
    dropout=0.1,       # Низкий dropout
    ff_dim=64,         # Средний размер
    n_block=2,
)

# Модель 2: Средняя
model_2 = TSMixerx(
    h=6,
    input_size=48,
    n_series=n_wells,
    dropout=0.15,      # Средний dropout
    ff_dim=128,        # Больший размер
    n_block=2,
)

# Модель 3: Агрессивная (больше capacity)
model_3 = TSMixerx(
    h=6,
    input_size=48,
    n_series=n_wells,
    dropout=0.2,       # Высокий dropout
    ff_dim=64,         # Средний размер
    n_block=3,         # Больше блоков
)

# Создать ensemble
ensemble = EnsembleForecaster(
    models=[model_1, model_2, model_3],
    mode="weighted",
    weights=[0.4, 0.3, 0.3],  # Больше веса на консервативную модель
)
```

**Преимущества:**
- ✅ Очень просто интегрировать
- ✅ Все модели имеют одинаковый интерфейс
- ✅ Быстро обучается
- ✅ Проверенный подход

**Ожидаемый эффект:** +4-6%

---

### Стратегия 2: Different Architectures (СРЕДНЯЯ СЛОЖНОСТЬ) ⭐⭐

**Идея:** Комбинировать разные типы архитектур

**Модели:**
```python
from neuralforecast.models import TSMixerx
from src.models_advanced import MultiScaleTSMixer

# Модель 1: Классический TSMixerx
model_1 = TSMixerx(
    h=6,
    input_size=48,
    n_series=n_wells,
    dropout=0.1,
    ff_dim=64,
)

# Модель 2: TSMixerx с другими параметрами
model_2 = TSMixerx(
    h=6,
    input_size=48,
    n_series=n_wells,
    dropout=0.15,
    ff_dim=128,
)

# Модель 3: MultiScale (новая архитектура)
model_3 = MultiScaleTSMixer(
    input_size=48,
    horizon=6,
    n_series=n_wells,
    scales=[1, 2, 4],     # Три масштаба
    hidden_dim=64,
    n_blocks=2,
    dropout=0.1,
)

# Ensemble
ensemble = EnsembleForecaster(
    models=[model_1, model_2, model_3],
    mode="weighted",
    weights=[0.35, 0.35, 0.30],  # Почти равные веса
)
```

**Преимущества:**
- ✅ Больше diversity - разные архитектуры
- ✅ MultiScale видит другие паттерны
- ✅ Лучшая обобщающая способность

**Ожидаемый эффект:** +6-8%

---

### Стратегия 3: Mixed Training Seeds (DIVERSITY ЧЕРЕЗ RANDOM INIT) ⭐⭐

**Идея:** Одинаковые модели, но разные начальные веса

**Модели:**
```python
import torch

# Одинаковые параметры, но разные random seeds
models = []
for seed in [42, 123, 456]:
    torch.manual_seed(seed)
    
    model = TSMixerx(
        h=6,
        input_size=48,
        n_series=n_wells,
        dropout=0.12,
        ff_dim=64,
        n_block=2,
    )
    models.append(model)

# Ensemble с равными весами
ensemble = EnsembleForecaster(
    models=models,
    mode="average",  # Простое среднее
)
```

**Преимущества:**
- ✅ Просто реализовать
- ✅ Diversity через разную инициализацию
- ✅ Защита от "плохих" случайных инициализаций

**Ожидаемый эффект:** +3-5%

---

### Стратегия 4: Feature Subsets (ADVANCED) ⭐

**Идея:** Разные модели видят разные наборы признаков

**Модели:**
```python
# Модель 1: Все признаки
features_all = [
    "wlpt", "womt", "womr", "wbhp", "wwir", "wwit",
    "inj_wwir_lag_weighted", 
    "wlpr_x_wbhp", "wlpr_x_inj_wwir_lag_weighted",
    "wlpr_ma3", "wlpr_ma6", "wlpr_ma12",
    # ... все 38 признаков
]

model_1 = TSMixerx(...)  # Использует features_all

# Модель 2: Только физические признаки
features_physics = [
    "wlpt", "womt", "womr", "wbhp", "wwir", "wwit",
    "inj_wwir_lag_weighted",
]

model_2 = TSMixerx(...)  # Использует features_physics

# Модель 3: Только engineered признаки
features_engineered = [
    "wlpr_x_wbhp", "wlpr_x_inj_wwir_lag_weighted",
    "wlpr_ma3", "wlpr_ma6", "wlpr_ma12",
    "wbhp_ma3", "wbhp_ma6", "wbhp_ma12",
]

model_3 = TSMixerx(...)  # Использует features_engineered

# Ensemble
ensemble = EnsembleForecaster(
    models=[model_1, model_2, model_3],
    mode="weighted",
    weights=[0.5, 0.25, 0.25],  # Больше веса на полную модель
)
```

**Преимущества:**
- ✅ Очень высокая diversity
- ✅ Защита от переобучения на конкретных признаках
- ✅ Интерпретируемость (какие признаки важнее)

**Недостатки:**
- ⚠️ Сложнее реализовать (нужно фильтровать признаки)
- ⚠️ Больше кода

**Ожидаемый эффект:** +5-7%

---

### Стратегия 5: Temporal Splitting (ADVANCED) ⭐

**Идея:** Модели специализируются на разных горизонтах

**Модели:**
```python
# Модель 1: Короткий горизонт (1-3 месяца)
model_short = TSMixerx(
    h=3,  # Только 3 месяца вперед
    input_size=24,  # Меньше истории
    n_series=n_wells,
    dropout=0.1,
    ff_dim=64,
)

# Модель 2: Средний горизонт (4-6 месяцев)
model_mid = TSMixerx(
    h=6,
    input_size=36,  # Средняя история
    n_series=n_wells,
    dropout=0.12,
    ff_dim=96,
)

# Модель 3: Длинный горизонт (весь прогноз)
model_long = TSMixerx(
    h=6,
    input_size=48,  # Полная история
    n_series=n_wells,
    dropout=0.15,
    ff_dim=128,
)

# Комбинирование с весами по горизонту
# (требует custom логики в EnsembleForecaster)
```

**Преимущества:**
- ✅ Каждая модель оптимизирована для своего горизонта
- ✅ Лучше для разных типов прогнозов

**Недостатки:**
- ⚠️ Сложная интеграция
- ⚠️ Нужна модификация EnsembleForecaster

**Ожидаемый эффект:** +4-6%

---

## 🎯 РЕКОМЕНДУЕМАЯ СТРАТЕГИЯ ДЛЯ СТАРТА

### ✅ Начните со Стратегии 1 + элементы Стратегии 2

**Почему:** Простота + эффективность

**Конкретная конфигурация:**

```python
from neuralforecast.models import TSMixerx
from src.models_advanced import EnsembleForecaster, MultiScaleTSMixer

def create_ensemble_model(config, n_series):
    """Create ensemble with 4 diverse models."""
    
    base_config = {
        'h': config.horizon,
        'input_size': config.input_size,
        'n_series': n_series,
        'hist_exog_list': config.hist_exog,
        'futr_exog_list': config.futr_exog,
        'stat_exog_list': config.static_exog,
    }
    
    # Модель 1: Консервативная TSMixerx
    model_1 = TSMixerx(
        **base_config,
        dropout=0.08,
        ff_dim=64,
        n_block=2,
        revin=True,
    )
    
    # Модель 2: Средняя TSMixerx
    model_2 = TSMixerx(
        **base_config,
        dropout=0.12,
        ff_dim=96,
        n_block=2,
        revin=True,
    )
    
    # Модель 3: Агрессивная TSMixerx
    model_3 = TSMixerx(
        **base_config,
        dropout=0.18,
        ff_dim=128,
        n_block=3,
        revin=True,
    )
    
    # Модель 4: MultiScale (если Phase 2)
    model_4 = MultiScaleTSMixer(
        input_size=config.input_size,
        horizon=config.horizon,
        n_series=n_series,
        scales=[1, 2, 4],
        hidden_dim=64,
        n_blocks=2,
        dropout=0.1,
    )
    
    # Создать ensemble
    ensemble = EnsembleForecaster(
        models=[model_1, model_2, model_3, model_4],
        mode="weighted",
        weights=[0.30, 0.25, 0.20, 0.25],  # Balanced
    )
    
    return ensemble
```

**Эта конфигурация дает:**
- ✅ Diversity через разные гиперпараметры (модели 1-3)
- ✅ Diversity через разные архитектуры (модель 4)
- ✅ Простая интеграция
- ✅ **Ожидаемый эффект: +6-8%**

---

## 📊 Сравнение Стратегий

| Стратегия | Сложность | Эффект | Diversity | Время обучения |
|-----------|-----------|--------|-----------|----------------|
| **1. TSMixerx variations** | ⭐ | +4-6% | ⭐⭐ | 1x |
| **2. Different architectures** | ⭐⭐ | +6-8% | ⭐⭐⭐ | 1.2x |
| **3. Random seeds** | ⭐ | +3-5% | ⭐⭐ | 1x |
| **4. Feature subsets** | ⭐⭐⭐ | +5-7% | ⭐⭐⭐ | 1x |
| **5. Temporal splitting** | ⭐⭐⭐ | +4-6% | ⭐⭐ | 1.1x |
| **Комбинированная (1+2)** | ⭐⭐ | **+6-8%** | ⭐⭐⭐ | **1.2x** |

---

## 🎨 3 Режима Комбинирования

### 1. Average (Простое среднее)
```python
ensemble = EnsembleForecaster(
    models=[model_1, model_2, model_3],
    mode="average",
)
```

**Когда использовать:** Модели примерно одинаковой точности

### 2. Weighted (Взвешенное среднее)
```python
ensemble = EnsembleForecaster(
    models=[model_1, model_2, model_3],
    mode="weighted",
    weights=[0.5, 0.3, 0.2],  # Больше веса на лучшую модель
)
```

**Когда использовать:** Знаете, какая модель лучше (из валидации)

**Как выбрать веса:**
```python
# 1. Обучить модели по отдельности
# 2. Проверить на валидации
# 3. Установить веса пропорционально точности

val_mae_1 = 12.5
val_mae_2 = 13.8
val_mae_3 = 14.2

# Инвертировать (меньше MAE = больше вес)
inv_errors = [1/val_mae_1, 1/val_mae_2, 1/val_mae_3]
total = sum(inv_errors)
weights = [x/total for x in inv_errors]
# weights ≈ [0.40, 0.36, 0.24]
```

### 3. Stacking (Мета-модель)
```python
ensemble = EnsembleForecaster(
    models=[model_1, model_2, model_3],
    mode="stacking",  # Обучает мета-модель
)
```

**Когда использовать:** Максимальная точность, есть время

**Как работает:** Нейронная сеть учится оптимально комбинировать прогнозы

---

## 💡 Практические Советы

### 1. Начните с 3 моделей
- Меньше - мало diversity
- Больше - медленнее, diminishing returns
- **3-4 модели = оптимум**

### 2. Проверьте diversity
```python
# Корреляция между ошибками моделей должна быть НИЗКОЙ
import numpy as np

errors_1 = y_true - predictions_1
errors_2 = y_true - predictions_2

correlation = np.corrcoef(errors_1, errors_2)[0, 1]

# Хорошо: correlation < 0.7 (модели ошибаются по-разному)
# Плохо: correlation > 0.9 (модели почти идентичны)
```

### 3. Валидация весов
```python
# Используйте walk-forward CV для выбора весов
best_weights = None
best_mae = float('inf')

for w1 in [0.2, 0.3, 0.4, 0.5]:
    for w2 in [0.2, 0.3, 0.4]:
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue
        
        # Протестировать на валидации
        ensemble.weights = torch.tensor([w1, w2, w3])
        mae = evaluate_on_validation(ensemble)
        
        if mae < best_mae:
            best_mae = mae
            best_weights = [w1, w2, w3]
```

### 4. Мониторинг обучения
```python
# Следите за тем, что все модели обучаются
# Если одна модель "застряла" - перезапустите с другим random seed

for i, model in enumerate(ensemble.models):
    train_loss = model.get_train_loss()
    print(f"Model {i}: train_loss = {train_loss}")
    
    # Если loss не уменьшается - проблема
    if train_loss > 100:
        print(f"WARNING: Model {i} not converging!")
```

---

## ✅ Пошаговая Интеграция

### Шаг 1: Создайте helper функцию
```python
# В wlpr_pipeline.py

def _create_single_tsmixer(config, n_series, **kwargs):
    """Helper to create single TSMixerx with custom params."""
    
    base_params = {
        'h': config.horizon,
        'input_size': config.input_size,
        'n_series': n_series,
        'hist_exog_list': config.hist_exog,
        'futr_exog_list': config.futr_exog,
        'stat_exog_list': config.static_exog,
        'revin': config.revin,
        'scaler_type': config.scaler_type,
        'n_block': config.n_block,
        'ff_dim': config.ff_dim,
        'dropout': config.dropout,
    }
    
    # Override with custom kwargs
    base_params.update(kwargs)
    
    return TSMixerx(**base_params)
```

### Шаг 2: Модифицируйте _create_model()
```python
def _create_model(config, n_series):
    """Create model (single or ensemble)."""
    
    if config.model_type == "ensemble":
        from models_advanced import EnsembleForecaster
        
        # Create diverse models
        models = [
            _create_single_tsmixer(config, n_series, dropout=0.08, ff_dim=64),
            _create_single_tsmixer(config, n_series, dropout=0.12, ff_dim=96),
            _create_single_tsmixer(config, n_series, dropout=0.18, ff_dim=128),
        ]
        
        # Add MultiScale if enabled
        if config.use_multiscale_in_ensemble:
            from models_advanced import MultiScaleTSMixer
            multiscale = MultiScaleTSMixer(...)
            models.append(multiscale)
        
        # Create ensemble
        model = EnsembleForecaster(
            models=models,
            mode=config.ensemble_mode,
            weights=config.ensemble_weights,
        )
        
        return model
    
    else:
        # Single model (existing code)
        return _create_single_tsmixer(config, n_series)
```

### Шаг 3: Добавьте в PipelineConfig
```python
@dataclass
class PipelineConfig:
    # ... existing params ...
    
    # Ensemble configuration
    model_type: str = "single"  # "single" or "ensemble"
    ensemble_mode: str = "weighted"  # "average", "weighted", "stacking"
    ensemble_weights: Optional[List[float]] = None
    use_multiscale_in_ensemble: bool = False
```

---

## 🎯 Итоговая Рекомендация

**Для Phase 2 используйте:**

✅ **4 модели в ensemble:**
1. TSMixerx (dropout=0.08, ff_dim=64) - консервативная
2. TSMixerx (dropout=0.12, ff_dim=96) - средняя
3. TSMixerx (dropout=0.18, ff_dim=128) - агрессивная
4. MultiScaleTSMixer (scales=[1,2,4]) - multi-scale

✅ **Режим:** weighted average

✅ **Веса:** [0.30, 0.25, 0.20, 0.25] (подобрать на валидации)

**Ожидаемый результат:** +6-8% над Phase 1

---

**Готово к интеграции!** 🚀
