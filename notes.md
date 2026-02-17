Анализ 4 статей и сверка с нашим пайплайном

   Статья 1: PIGNN -- Physics-Informed GNN for Fluid Flow (Chen et al., 2025)

   Суть: Encode-process-decode GNN для предсказания давления в пористой среде. Физический граф-конволюционный
    kernel встраивает дискретизированное уравнение Дарси прямо в message passing. Двухфазное обучение:
   сначала data loss, потом data+physics loss.

   Что можно применить у нас:
   •  Ничего напрямую. Эта статья решает задачу pressure field simulation на PEBI-сетках (замена
      CMG/Eclipse). Наш пайплайн -- прогноз добычи, а не симуляция давления. Chronos-2 -- frozen foundation
      model, мы не можем встроить custom graph convolution kernel.

   ──────────────────────────────────────────

   Статья 2: SGP-GCN (Liu et al., 2025, Energy Engineering)

   Суть: GCN с пространственно-геологической матрицей смежности для multi-well прогноза. Ключевые идеи:
   1. Spatial-Geological Adjacency Matrix = нормализованное расстояние + косинусная схожесть геологических
      параметров (пористость, проницаемость, насыщенность, толщина, пластовое давление)
   2. Production Clustering (SPC) -- K-Means кластеризация скважин по профилю добычи, разреживание матрицы
      смежности (удаление связей между кластерами с низким вниманием)
   3. Multi-Scale Dilated CNN + LSTM + Self-Attention для temporal features
   4. MDI (Random Forest) для отбора динамических фич

   Что можно применить у нас:

   Идея                             │ Применимость                     │ Статус у нас
   ---------------------------------+----------------------------------+---------------------------------
   Геологическая схожесть в граф    │ **Да, частично** -- у нас есть   │ Граф строится только по
                                    │ coord_x/y/z, но нет              │ расстоянию (IDW)
                                    │ пористости/проницаемости/толщины │
                                    │ в данных                         │
   K-Means кластеризация скважин +  │ **Да** -- может убрать шумные    │ Нет, граф полносвязный (k=5
   разреживание графа               │ связи между скважинами с разным  │ ближайших, но без фильтрации по
                                    │ поведением (напр. Well 14 vs     │ поведению)
                                    │ остальные)                       │
   MDI feature selection            │ **Да** -- Random Forest feature  │ Мы делали ручной анализ, но не
                                    │ importance как дополнение к      │ автоматический отбор
                                    │ нашему Pearson/MI анализу        │
   Multi-Scale Dilated CNN          │ **Нет** -- Chronos-2 frozen, не  │ N/A
                                    │ можем менять архитектуру         │

   ──────────────────────────────────────────

   Статья 3: STA-MGCN (Lu et al., 2025, Scientific Reports)

   Суть: Multi-graph GCN с 4 разными графами + adaptive attention fusion. Самая релевантная статья.

   4 графа:
   1. G_topo -- геометрическая топология (кратчайший путь)
   2. G_bin -- бинарная связь нагнетание-добыча (0/1)
   3. G_cond -- гидропроводность (harmonic mean permeability / distance) -- учитывает гетерогенность
   4. G_dyn -- DTW (Dynamic Time Warping) между кривыми изменения добычи + Gaussian kernel

   Ключевые находки авторов: G_cond и G_dyn дали наибольший вклад. Удаление G_cond увеличило RMSE на 235%,
   G_dyn -- на 175%. Adaptive fusion превзошла fixed-weight на 103%.

   Что можно применить у нас:

   Идея                             │ Применимость                     │ Детали
   ---------------------------------+----------------------------------+---------------------------------
   **DTW-based Dynamic Similarity   │ **Высокая**                      │ Вместо одного статического графа
   Graph**                          │                                  │ (по расстоянию), добавить
                                    │                                  │ динамический граф на основе DTW
                                    │                                  │ между кривыми добычи. DTW
                                    │                                  │ схожесть -> Gaussian kernel ->
                                    │                                  │ весовая матрица для neighbor
                                    │                                  │ aggregation. У нас
                                    │                                  │ `neighbor_avg_wlpr` использует
                                    │                                  │ только geographic proximity --
                                    │                                  │ DTW-based aggregation учтет, что
                                    │                                  │ далекие скважины могут вести
                                    │                                  │ себя похоже
   **Injection-Production Binary    │ **Частично есть**                │ Наш CRM connectivity по сути это
   Graph**                          │                                  │ аналог G_bin, но мы используем
                                    │                                  │ его как скалярную фичу
                                    │                                  │ (`crm_max_connectivity`), а не
                                    │                                  │ как граф для message passing
   **Fluid Conductance Graph**      │ **Нет данных**                   │ Требует проницаемость пласта --
                                    │                                  │ у нас нет
   **Adaptive Multi-Graph Fusion**  │ **Нет** -- Chronos-2 не          │ Но можно сделать **два набора
                                    │ поддерживает attention fusion    │ neighbor_avg**: один по
                                    │                                  │ расстоянию, другой по DTW -- и
                                    │                                  │ дать оба как ковариаты
   **Hybrid temporal sequences**    │ **Уже есть**                     │ Наши hist_exog (лаговые) +
   (lag features + future           │                                  │ futr_exog (future injection) --
   controllable)                    │                                  │ аналогичный подход

   ──────────────────────────────────────────

   Статья 4: PI-GNN (Liu & Pyrcz, UT Austin)

   Суть: Physics-Informed GNN для waterflood production forecasting. GCN с bipartite граф (injector ->
   producer). CRM material balance как physics loss.

   Ключевые техники:
   1. CRM differential form as physics loss: (Ct*Vp/J)*dq/dt + q + dPwf/dt*C*Vp - I*F = 0
   2. Trainable adjacency matrix -- connectivity F обучается, а не задается
   3. Unsupervised learning of time-dependent J(t) (productivity index) and Vp(t) (drainage volume) --
      relaxing CRM assumptions
   4. Fast Marching Method для domain-informed adjacency matrix

   Что можно применить у нас:

   Идея                             │ Применимость                     │ Детали
   ---------------------------------+----------------------------------+---------------------------------
   **CRM-based physics loss**       │ **Нет** -- Chronos-2 frozen, нет │ Но если перейдем на fine-tuning
                                    │ custom loss                      │ в будущем, это top candidate
   **Fast Marching adjacency**      │ **Нет** -- нужна карта           │ У нас нет геологической модели
                                    │ проницаемости                    │
   **CRM connectivity как           │ **Уже есть**                     │ `crm_max_connectivity` в
   ковариата**                      │                                  │ futr_exog
   **Time-dependent productivity    │ **Интересно**                    │ Можно вычислить J(t) = q(t) /
   index J(t)**                     │                                  │ (Pe - Pwf(t)) как дополнительную
                                    │                                  │ фичу. У нас есть wlpr и wthp
                                    │                                  │ (wellhead pressure). Это
                                    │                                  │ физически осмысленная
                                    │                                  │ производная фича

   ──────────────────────────────────────────

   Сверка с интернетом: что еще актуально (2025-2026)

   ChronosX (Amazon, March 2025)

   Модульный метод инъекции ковариат в Chronos без изменения core модели. Создает отдельный covariate
   encoder, выход которого суммируется с embeddings основной модели. Пока не интегрирован в Darts, но
   показывает направление -- наш подход (ковариаты через past_covariates/future_covariates API Darts)
   корректен и является стандартным.

   Augmented PINNs для waterflood (SPE Journal, 2025)

   Augmented PINNs превзошли CRM в прогнозе для заводнения. Ключевой вывод: physics constraints критически
   важны для точности. Подтверждает, что наш CRM connectivity -- правильная идея, но мы используем его слабо
   (только как скалярную фичу).

   MathAI 2025: GraphSAGE + LSTM для межскважинного влияния

   R^2=0.97 на прогнозе взаимного влияния скважин. Подтверждает: graph message passing между скважинами --
   state-of-the-art. Наш neighbor_avg_* -- упрощенная версия этого подхода.

   ──────────────────────────────────────────

   Итого: конкретные улучшения, которые мы можем реализовать

   Ранжировано по ожидаемому эффекту и сложности:

   1. DTW-based neighbor aggregation (из STA-MGCN)
   •  Что: Вычислить DTW distance между кривыми добычи всех пар скважин. Построить similarity matrix
      (Gaussian kernel). Использовать её для второго набора neighbor_avg_* features.
   •  Зачем: Наш текущий neighbor_avg_wlpr использует только географическую близость. DTW учтет, что скважины
       с похожей динамикой добычи (даже далекие) могут быть лучшими "соседями" для прогноза.
   •  Сложность: Средняя (dtw-python/tslearn уже есть). ~100 строк кода.
   •  Источник: STA-MGCN (Paper 3), подтверждено web search.

   2. Production clustering + graph sparsification (из SGP-GCN)
   •  Что: K-Means кластеризация скважин по профилю добычи. Разрезать связи между скважинами из разных
      кластеров в графе.
   •  Зачем: Well 14 (4.8 m3/d) сильно отличается от соседей (146-166 m3/d). Кластеризация выделит его в
      отдельный кластер и ослабит шумное влияние "нетипичных" соседей.
   •  Сложность: Низкая. ~50 строк.
   •  Источник: SGP-GCN (Paper 2).

   3. Productivity Index J(t) как фича (из PI-GNN)
   •  Что: J(t) = wlpr(t) / max(wthp_ref - wthp(t), epsilon). Физически осмысленная производная фича --
      "способность скважины давать продукцию при данном давлении".
   •  Зачем: Ловит изменения в продуктивности скважины, которые Chronos-2 не может вывести из raw wlpr/wthp
      по отдельности.
   •  Сложность: Низкая. ~20 строк.
   •  Источник: PI-GNN (Paper 4).

   4. MDI Feature Selection (из SGP-GCN)
   •  Что: Random Forest MDI (Mean Decrease Impurity) для автоматического отбора фич вместо ручного анализа.
      Убрать фичи с MDI < threshold.
   •  Зачем: Систематическая замена нашего ручного Pearson/MI анализа. Может обнаружить нелинейные
      зависимости, которые MI пропускает.
   •  Сложность: Низкая. ~30 строк (sklearn RF feature_importances_).
   •  Источник: SGP-GCN (Paper 2).

   5. Bilateral Gaussian Filter вместо Savitzky-Golay (из STA-MGCN)
   •  Что: Adaptive bilateral Gaussian filter для preprocessing вместо нашего Savitzky-Golay. Лучше сохраняет
       резкие изменения (shutdowns, startups) при сглаживании шума.
   •  Зачем: Savitzky-Golay сглаживает все одинаково. Bilateral filter адаптивен -- сильнее сглаживает
      плавные участки, слабее -- резкие перепады.
   •  Сложность: Низкая. ~30 строк.
   •  Источник: STA-MGCN (Paper 3).

   Не применимо (и почему):
   •  Custom GNN архитектуры (все 4 статьи) -- Chronos-2 frozen, нельзя заменить/дополнить encoder
   •  Physics loss (Papers 1, 4) -- нет custom training loop
   •  Trainable adjacency matrix (Paper 4) -- нет обучаемых параметров в Chronos-2 zero-shot
   •  Fast Marching / перmeability map (Paper 4) -- нет геологических данных
   •  Multi-Scale Dilated CNN (Paper 2) -- архитектура фиксирована