# Техническая спецификация интеграции GNN и TSMixerx

На основе предоставленной документации `TSMixerx`, мы детализируем архитектуру "Сэндвич". Мы не будем переписывать TSMixer с нуля, а используем его возможности по работе с экзогенными переменными (`exog`), чтобы внедрить результаты работы графовой сети.

## 1. Концепция "GNN как Генератор Фичей"

В документации `TSMixerx` указано:
> **TSMixerx:** MLP-based multivariate forecasting that combines temporal–feature mixing with **static** and **future** covariate support.

Мы используем GNN как "умный препроцессор", который превращает сырые данные соседей в **исторические экзогенные признаки** (`hist_exog`) для TSMixerx.

### Поток данных (Data Flow):

1.  **Вход:** Тензор $X_{raw}$ (история добычи всех скважин).
2.  **GNN Block (Spatial):**
    *   Проходит по графу связей.
    *   Агрегирует состояние соседей.
    *   **Выход:** Вектор $E_{gnn}$ (Embedding соседей).
3.  **Инъекция:** Мы подаем $E_{gnn}$ в TSMixerx как `hist_exog` (историческая экзогенная переменная).
4.  **TSMixerx (Temporal):**
    *   Принимает свою историю ($Y_{target}$).
    *   Принимает "мнение соседей" ($E_{gnn}$) через канал `hist_exog`.
    *   Смешивает их через MLP-слои.
5.  **Выход:** Прогноз $\hat{Y}$.

---

## 2. Реализация на уровне кода (Subclassing)

Мы создадим класс `GraphTSMixer`, наследующий от `TSMixerx`. Это позволит сохранить всю логику обучения, лоссов и валидации `NeuralForecast`, добавив только графовый слой.

```python
import torch
from torch import nn
from neuralforecast.models import TSMixerx
# Предполагаем наличие PyG
from torch_geometric.nn import GATv2Conv

class GraphTSMixer(TSMixerx):
    def __init__(self, n_series, input_size, h, graph_edge_index, gnn_hidden=32, **kwargs):
        """
        Args:
            graph_edge_index: (2, num_edges) тензор связей графа (static).
            kwargs: параметры для TSMixerx (n_block, ff_dim, dropout, etc.)
        """
        # Инициализируем родительский TSMixerx
        super().__init__(n_series=n_series, input_size=input_size, h=h, **kwargs)
        
        # Сохраняем граф (он статический в рамках батча)
        self.register_buffer('edge_index', graph_edge_index)
        
        # --- GNN LAYER ---
        # Вход: 1 фича (добыча wlpr) или больше. Выход: gnn_hidden фичей
        self.gnn = GATv2Conv(in_channels=1, out_channels=gnn_hidden, heads=1)
        
        # Адаптер для объединения фичей перед TSMixer
        # TSMixer ожидает: input_size (lags) + hist_exog + stat_exog
        # Мы добавляем gnn_hidden каналов к hist_exog
        self.gnn_projection = nn.Linear(gnn_hidden, gnn_hidden)

    def forward(self, windows_batch):
        """
        Переопределяем forward, чтобы внедрить GNN перед TSMixer.
        windows_batch - словарь от NeuralForecast (insample_y, hist_exog, etc.)
        """
        # 1. Достаем сырую историю [Batch, Input_Size, N_Series] 
        # (NeuralForecast обычно батчит по окнам, тут нужно аккуратно с размерностями)
        # Для упрощения: допустим, мы работаем с полным графом в батче
        x_in = windows_batch['insample_y'] 
        
        # 2. GNN Pass (Space)
        # GNN работает по узлам. Нам нужно прогнать GNN для каждого временного шага
        # или использовать Temporal GNN.
        # Упрощенно: берем среднее или последний шаг для контекста
        
        # [Batch, N_Nodes, Features]
        gnn_out = self.gnn(x_in, self.edge_index) 
        
        # 3. Инъекция в hist_exog
        # Добавляем выход GNN к существующим историческим экзогенным переменным
        if 'hist_exog' in windows_batch:
            windows_batch['hist_exog'] = torch.cat([windows_batch['hist_exog'], gnn_out], dim=-1)
        else:
            windows_batch['hist_exog'] = gnn_out
            
        # 4. Запускаем стандартный TSMixerx (Time)
        # Он сам разберется с Mixing Layers, используя обновленный hist_exog
        return super().forward(windows_batch)
```

## 3. Почему это сработает?

1.  **Совместимость:** TSMixerx уже умеет работать с `hist_exog_list`. Для него выход GNN — это просто еще одна "переменная", как давление или температура, только синтетическая.
2.  **Мощь TSMixer:** Параметры `n_block`, `ff_dim`, `dropout` из документации позволят настроить глубину временного анализа.
3.  **Revin (Reverse Instance Normalization):** Параметр `revin=True` в TSMixerx критически важен. Он нормализует входные ряды, чтобы GNN и MLP не сходили с ума от разного масштаба дебитов (10 кубов vs 500 кубов).

## 4. Конфигурация модели (Пример)

```python
model = GraphTSMixer(
    h=6,                        # Горизонт прогноза (6 месяцев)
    input_size=24,              # История (24 месяца)
    n_series=50,                # Количество скважин
    graph_edge_index=edges,     # Наш граф (1000м + геология)
    
    # Параметры TSMixerx
    n_block=4,                  # Глубина миксера
    ff_dim=64,                  # Ширина слоев
    dropout=0.1,
    revin=True,                 # Включить нормализацию
    loss=AdaptivePhysicsLoss()  # Наш кастомный лосс
)
```

Этот подход превращает сложную задачу в конструктор Lego: берем стандартный кубик TSMixerx и приделываем к нему "антенну" GNN для связи с соседями.
