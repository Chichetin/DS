# ML4Pay — предсказание возвратов в Avito Доставке

> Хакатон ITMO × Avito, команда **«ИИ в массы»**. Дедлайн: **10.05.2026 23:59**.

## TL;DR

Решаем задачу бинарной классификации: для каждой пары `(item_id, deliveryorder_id)` в test-выборке (4 030 995 строк за 01-10.10.2025) предсказать `is_return` (вернёт покупатель товар или нет). Базовая частота возвратов в данных — **4.56%** (сильный дисбаланс).

Текущий лидер на holdout (последние 10 дней train, 2025-09-21..09-30):

| Модель | ROC-AUC | best F1 | precision | recall |
|---|---|---|---|---|
| LGBM C1 (24 фичи) | 0.76741 | 0.22468 | 0.16 | 0.38 |
| LGBM C2 (+history) | 0.79584 | 0.27506 | 0.22 | 0.38 |
| LGBM C3 (+microcat/city TE) | 0.79702 | 0.27642 | 0.23 | 0.34 |
| LGBM C4 (+seller TE, dynamics, price ratio, is_first_sale) | 0.79698 | 0.27534 | 0.23 | 0.34 |
| **CatBoost C4** (на 15% сэмпле) | 0.80466 | 0.28160 | 0.23 | 0.36 |
| **Blend LGBM C4 + CatBoost C4** (α=0.30) | 0.80492 | 0.28199 | 0.235 | 0.35 |
| **LGBM full** (на ПОЛНОМ train, 29.15M строк) | **0.80496** | 0.27980 | 0.226 | 0.367 |
| **CatBoost-22** (на 22% сэмпле) | _идёт прогон_ | _—_ | _—_ | _—_ |
| **🏁 Финальный blend (LGBM-full + CatBoost-22)** | _обновится после прогона_ | _—_ | _—_ | _—_ |

> **Важно:** ROC-AUC и F1 у нас оценивается на **полном holdout** (~3.7M строк за 09-21..09-30, без сэмплинга). Метрики стабильны.

---

## Содержание

1. [Структура репо](#структура-репо)
2. [Что лежит в данных](#что-лежит-в-данных)
3. [Препроцессинг](#препроцессинг)
4. [Фичи и зачем они](#фичи-и-зачем-они)
5. [Модели и пайплайн](#модели-и-пайплайн)
6. [Как запустить с нуля](#как-запустить-с-нуля)
7. [Сабмит](#сабмит)
8. [Что осталось / куда копать команде](#что-осталось--куда-копать-команде)
9. [Окружение и подводные камни](#окружение-и-подводные-камни)
10. [Что попробовали и не выстрелило](#что-попробовали-и-не-выстрелило)

---

## Структура репо

```
DS/
├── data/
│   ├── raw/                    # исходники от организаторов (csv, .gitignore'd)
│   ├── clean/                  # после препроцессинга (parquet zstd, ~13 ГБ)
│   │   ├── orders_train.parquet           # 33 174 404 строк
│   │   ├── orders_test.parquet            # 4 030 995 строк
│   │   ├── orders_train_sample.parquet    # 15% buyers ≈ 4.97M (legacy)
│   │   ├── orders_train_sample22.parquet  # 22% buyers ≈ 7.3M (для CatBoost)
│   │   ├── items.parquet, users.parquet, payments.parquet, payments_agg.parquet
│   └── features/               # фича-таблицы (parquet zstd, ~15-20 ГБ)
├── src/                        # пайплайн (см. ниже)
├── artifacts/                  # логи, модели, метрики, предсказания
├── requirements.txt            # pip-зависимости (зафиксированные версии)
├── .gitignore
└── README.md                   # ты здесь
```

`data/`, `artifacts/`, venv (`Scripts/`, `Lib/`, `bin/`), `*.zip` — всё это в `.gitignore`. После клона надо самому собрать venv (см. ниже) и положить раздачу `ITMO_Avito_Hackathon.zip` от организаторов в корень. Дальше пайплайн всё восстанавливает.

---

## Что лежит в данных

Организаторы дали 5 csv-таблиц за период **01.07.2025 — 10.10.2025**:

| Файл | Строк (примерно) | О чём |
|---|---|---|
| `orders_train.csv` | 39M (после дедупа 33M) | заказы за 01.07–30.09, есть `is_return` |
| `orders_test.csv` | 4M | заказы за 01.10–10.10, нужно предсказать `is_return` |
| `items.csv` | 20.6M | объявления (категория, микрокатегория, дата создания/закрытия) |
| `users.csv` | 11.7M | пользователи (gender, iscompany, isblocked, дата регистрации) |
| `payments.csv` | 30M | платежи по заказам (методы оплаты, суммы, тайминг) |

### Ключевые особенности данных, которые надо знать

- **Уровень предсказания — `item_id`** (FAQ #2 ТЗ). В одном `deliveryorder_id` может быть несколько товаров — каждый рассматриваем отдельно.
- **Дисбаланс классов: 4.56% positive.** Любая модель должна это учитывать (`is_unbalance=True` у LGBM, `auto_class_weights='Balanced'` у CatBoost).
- **Только 1 попытка сабмита** (FAQ #7). Лидерборд закроют ДО дедлайна — нет обратной связи. Поэтому self-валидация (наш holdout) — единственный сигнал.
- **Очень много полей в `orders` ликают** (заполнены ПОСЛЕ `order_create_date`): `cancel_date`, `accept_date`, `terminal_*`, `late_pay_*`. Все они исключены в препроцессинге — см. `feedback_leak_checklist.md` в локальной memory.
- **Платежи (refunds) НЕ записаны как отрицательные суммы** — все amounts > 0, это agregat по карточным транзакциям (см. `07_check_payment_leak.py` для проверки).
- **`order_accept_date.max() = 2026-04-26`** — это реальные данные (очень долгие финализации заказов), не баг.

---

## Препроцессинг

Скрипт `src/02_preprocess.py` (запускается один раз, ~11 мин на 16 ГБ):

1. **Дедуп `orders`.** В исходных csv есть полные дубли строк (артефакт экспорта) — снимаем через DuckDB (polars OOMит на 39M).
2. **Time-split orders.** Train = `< 2025-10-01`, test = `2025-10-01..10-10` (по `order_create_date`).
3. **Удаляем ликающие поля.** Из `orders` оставляем только то, что было известно к моменту `order_create_date`.
4. **Нормализуем `payments`.** Метод оплаты «SBP» и «СБП» (один и тот же в кодировке cp1251) склеиваем.
5. **Аггрегируем `payments` до уровня заказа** (`payments_agg.parquet`): `n_pay`, `sum_amount`, `min_txtime`, `max_txtime`, `dominant_payment_method`. Эти аггрегаты — на уровне `deliveryorder_id`, чтобы дешевле джойнить.
6. **Считаем фичи `items`** (`is_active`, `lifetime_days`) и `users` (`is_seller`, `tenure_days`).

Все артефакты пишутся в `data/clean/` в parquet+zstd.

> **Зачем DuckDB, а не polars?** На больших groupby/unique polars даже в streaming-режиме на 16 ГБ ловит OOM с exit code 5. DuckDB с `PRAGMA memory_limit='8GB'` и spill в `artifacts/preprocess/duckdb_tmp/` отрабатывает стабильно.

---

## Фичи и зачем они

Мы строим фичи итерациями: C1 → C2 → C3 → C4. Каждый уровень добавляет блок и проверяется отдельным baseline'ом, чтобы видеть прирост.

### C1 — базовые фичи без истории

> Скрипт: `05_features_c1.py`. **Прирост: AUC 0.767, F1 0.225** (от голого baseline-majority это +0.225 F1 пункт).

24 фичи из самих заказов и справочников:

| Блок | Фичи | Зачем |
|---|---|---|
| Order temporal | `order_dow`, `order_dom`, `is_weekend` | сезонность дней недели — ТЗ FAQ #3 говорит «небольшой эффект может быть» |
| Order numeric | `order_price`, `log_order_price` | дорогие товары возвращают чаще (типичная закономерность ритейла) |
| Order categorical | `delivery_service`, `platform_id`, `city` | сильно разный поведенческий profile у разных способов доставки и регионов |
| Item | `category_name`, `microcat_name`, `item_lifetime_at_order`, `is_active_at_order` | категория сильно влияет на возвратность (одежда vs техника), активность объявления — proxy на «свежесть» |
| Buyer | `buyer_gender`, `buyer_iscompany`, `buyer_isblocked`, `buyer_tenure_days`, `buyer_is_seller` | новый покупатель vs старый, юрлицо vs физ |
| Seller | `seller_iscompany`, `seller_isblocked`, `seller_tenure_days` | продавец-новичок vs опытный |
| Payment | `n_pay`, `pay_sum_amount`, `dominant_payment_method`, `pay_to_price_ratio` | количество транзакций (split-payments часто = неуверенный покупатель), факт переплаты |

### C2 — buyer/seller/item history (ДО даты заказа)

> Скрипты: `08_history.py` (DuckDB rolling) → `09_features_c2.py` (join). **Прирост над C1: +0.028 AUC, +0.050 F1.** Это самый жирный шаг.

**Идея:** «как этот покупатель/продавец/товар вели себя в прошлом?» — самая мощная сигнатура в задаче возвратов. Работаем строго без time-leak: для каждого заказа берём агрегаты по `[-∞, order_create_date − 1]`.

Реализация: для каждой сущности (`buyer_id`, `seller_id`, `item_id`) считаем daily-агрегаты на ПОЛНОМ train (33M, не на сэмпле!), затем кумулятивные суммы через DuckDB `SUM() OVER (PARTITION BY E ORDER BY date)`. Из них получаем все нужные «past» метрики простым `cum - today_aggregate`.

| Блок (для каждой из 3 сущностей) | Фичи | Зачем |
|---|---|---|
| Counts | `*_past_orders`, `*_past_returns`, `*_past_30d_orders`, `*_past_7d_orders` | сколько уже было заказов и в каких окнах |
| Rates | `*_past_return_rate`, `*_past_30d_return_rate`, `*_past_7d_return_rate` | доля возвратов — главная сигнатура «возвратчик / нет» |
| Recency | `*_days_since_last_order` | как давно был последний заказ |
| Pricing | `*_past_avg_price` | какой средний чек, отклонение от него — аномалия |

Топ-фичи C2 в LGBM (gain): **`microcat_name`, `delivery_service`, `city`, `buyer_past_return_rate` (#4!), `n_pay`, `is_active_at_order`**. История buyer/seller прорывается в топ-10 — ровно то, чего и хотели.

### C3 — target encoding для микрокатегорий и городов

> Скрипты: `11_target_encoding.py` → `12_features_c3.py`. **Прирост над C2: +0.001 AUC** (LGBM научился из microcat_name сам), но **+0.005 AUC для CatBoost** (он любит явные сглаженные фичи).

Bayesian smoothing для категорий с длинным хвостом: вместо raw return-rate берём `(returns + α·μ) / (count + α)` где `α=100`, `μ=0.0456`. Без time-leak, окно `[-∞, date-1]`.

| Фича | Зачем |
|---|---|
| `microcat_te`, `microcat_te_count` | возвратность микрокатегории, сглаженная для редких |
| `city_te`, `city_te_count` | то же для города |

> 🤔 **Почему фича помогает CatBoost больше, чем LGBM?** LGBM учит дерево разбиений по `microcat_name` напрямую (split на категориальный код). CatBoost умеет ordered target stats, но с предобработкой получает явный численный сигнал и быстрее находит правильный split.

### C4 — продвинутые фичи

> Скрипты: `16_te_seller.py` → `17_features_c4.py`. **Прирост над C3: 0 для LGBM, ~0 для CatBoost** — модели уже выжали потолок текущих данных.

Несмотря на маленький прирост, эти фичи мы оставляем — они пригодятся для будущих экспериментов и для устойчивости к шуму.

| Фича | Зачем |
|---|---|
| `seller_te`, `seller_te_count` | TE для seller_id (high-cardinality, ~5M уникальных) |
| `microcat_median_price` | snapshot-медиана цены в микрокатегории |
| `price_to_microcat_median`, `log_price_to_microcat_median` | насколько товар дороже/дешевле медианы своей микрокат — аномально дешёвые могут возвращать чаще |
| `buyer_dynamics_30d_diff`, `buyer_dynamics_7d_diff` | отклонение свежей возвратности от исторической (стал ли байер чаще возвращать в последнее время) |
| `seller_dynamics_30d_diff`, `seller_dynamics_7d_diff` | то же для продавца |
| `is_first_sale` | первая ли это продажа этого item_id — обычно у новых товаров другая динамика |

В топе по importance из новых вошёл только **`seller_te`** (≈ #9). Остальные не дали явного сигнала, но и не ухудшили.

---

## Модели и пайплайн

### LightGBM (быстрый, для итераций)

```python
# src/18_baseline_c4.py
PARAMS = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    is_unbalance=True,            # критично для 4.5% дисбаланса
    n_estimators=1000, random_state=42,
)
# Категориальные передаются как pd.Categorical через categorical_feature=CAT_COLS
# Early stopping: 30-40 раундов на holdout AUC
```

**Время тренировки:** 1-3 мин на 4.36M (15% сэмпле) → 20-30 мин на полном train (33M, скрипт `25_train_lgbm_full.py` с Float32 даункастом).

### CatBoost (точнее, но медленнее)

```python
# src/19_baseline_catboost_c4.py
PARAMS = dict(
    iterations=2000, learning_rate=0.03, depth=6,
    eval_metric="AUC",
    auto_class_weights="Balanced",
    od_type="Iter", od_wait=30,
)
```

**Время тренировки:** 87 мин на 4.36M (15% сэмпле, depth=6) → ~3-4 ч на 22% сэмпле (7.3M). На полный train (33M) НЕ влезает в 16 ГБ RAM — нужен 64+ ГБ или GPU.

### Финальная модель — Blend

`src/27_blend_and_submit.py`:
1. Загружает holdout-вероятности от LGBM и CatBoost.
2. Перебирает `α ∈ [0, 1]` шагом 0.05, ищет такое, что максимизирует F1 на blendженом holdout.
3. Калибрует порог по F1.
4. Применяет (α, threshold) к test → бинаризация → CSV сабмит.

Текущий best на 15% сэмпле: **α = 0.30 (LGBM-вес), threshold ≈ 0.77, AUC 0.80492, F1 0.28199**.

---

## Как запустить с нуля

> Внимание: end-to-end ~3 ч на 16 ГБ RAM (без CatBoost — с CatBoost +3-4 ч). Все промежуточные данные на диске занимают ~30 ГБ.

**Windows / PowerShell:**

```powershell
# 0. Подготовка venv (Python 3.14)
python -m venv .
.\Scripts\activate
pip install -r requirements.txt

# 1. Распакуй ITMO_Avito_Hackathon.zip и положи csv-файлы в data/raw/
#    (orders.csv, items.csv, users.csv, payments.csv)

# 2. Препроцессинг + сэмплинг
$env:PYTHONIOENCODING = "utf-8"
.\Scripts\python.exe src/02_preprocess.py        # ~11 мин
.\Scripts\python.exe src/03_sanity.py            # быстрая проверка
.\Scripts\python.exe src/04_sample.py            # 15% сэмпл (legacy)
.\Scripts\python.exe src/24_resample_22.py       # 22% сэмпл (для CatBoost)

# 3. Глобальные фичи (агрегаты на ПОЛНОМ train, не зависят от сэмпла)
.\Scripts\python.exe src/08_history.py           # buyer/seller/item history
.\Scripts\python.exe src/11_target_encoding.py   # microcat/city TE
.\Scripts\python.exe src/16_te_seller.py         # seller TE

# 4. Фича-таблицы для каждого режима
# 4а. Полный train + полный holdout (для LGBM)
$env:SUFFIX = "_full"; $env:SAMPLE_PATH = "orders_train.parquet"; $env:TARGETS = "train,holdout"
.\Scripts\python.exe src/05_features_c1.py
.\Scripts\python.exe src/09_features_c2.py
.\Scripts\python.exe src/12_features_c3.py
.\Scripts\python.exe src/17_features_c4.py

# 4б. 22% сэмпл, только train (для CatBoost; holdout берёт _full)
$env:SUFFIX = "_22"; $env:SAMPLE_PATH = "orders_train_sample22.parquet"; $env:TARGETS = "train"
.\Scripts\python.exe src/05_features_c1.py
.\Scripts\python.exe src/09_features_c2.py
.\Scripts\python.exe src/12_features_c3.py
.\Scripts\python.exe src/17_features_c4.py

# 4в. Test (нужен один раз, не зависит от сэмпла)
Remove-Item Env:SUFFIX, Env:SAMPLE_PATH; $env:TARGETS = "test"
.\Scripts\python.exe src/05_features_c1.py
.\Scripts\python.exe src/09_features_c2.py
.\Scripts\python.exe src/12_features_c3.py
.\Scripts\python.exe src/17_features_c4.py

# 5. Тренировки
.\Scripts\python.exe src/25_train_lgbm_full.py    # ~30 мин
.\Scripts\python.exe src/26_train_catboost_22.py  # ~3-4 ч

# 6. Финальный blend + сабмит
.\Scripts\python.exe src/27_blend_and_submit.py `
    --lgbm artifacts/lgbm_full `
    --cat  artifacts/catboost_22 `
    --name final
```

Готовый сабмит: `artifacts/submission/final/submission.csv`.

**Linux / macOS:** то же самое, но:
- venv-папка называется `bin/`, активация `source bin/activate`,
- запуск через `python src/02_preprocess.py` (без `.\Scripts\python.exe`),
- env-переменные: `export SUFFIX=_full` вместо `$env:SUFFIX = "_full"`.

---

## Сабмит

**Формат (по ТЗ ML4Pay от Avito):**

| Колонка | Тип | Описание |
|---|---|---|
| `item_id` | string | id товара (как в test) |
| `deliveryorder_id` | string | id заказа доставки (одна и та же `item_id` может встречаться в нескольких заказах — это нормально) |
| `order_create_date` | date (YYYY-MM-DD) | дата создания заказа |
| `is_return` | bool | предсказание (true/false) |

Всего строк: **4 030 995** (по числу пар в test). CSV ~ 200-250 МБ.

**Куда отправлять:** `itmo.hack@itmo.ru`, тема — название команды **«ИИ в массы»**, в письме — приложить CSV.

**Дедлайн:** 10.05.2026 23:59.

---

## Что осталось / куда копать команде

### Прямо сейчас (если успеваем до дедлайна)

- [ ] **Финальный прогон on schedule** — `src/25/26/27` (см. выше). Метрики и сабмит-CSV — в `artifacts/submission/final/`.
- [ ] **Sanity-проверки сабмита перед отправкой:** target rate ≈ 4.5% (см. `metrics.txt`), нет NaN, все 4 030 995 строк уникальны по `(deliveryorder_id, item_id)`.
- [ ] **Перепроверить порог.** Стандартно ловим F1, но возможно лучше зафиксировать порог 0.5 или взять середину между F1-thr и Youden — это устойчивее на test.

### Если есть ещё время / для финальной презентации

- [ ] **Стэкинг.** Сейчас только blend (линейная комбинация). Стэкинг через мета-модель (LR на LGBM-prob + CatBoost-prob + несколько raw-фичей) обычно даёт +0.002-0.005 AUC.
- [ ] **CatBoost depth=7 на 22% сэмпле** — мы пробовали запустить (`src/23_catboost_c4_tune.py`), но не успели прогнать. На полных данных может выстрелить на +0.003-0.005 AUC.
- [ ] **Buyer-seller pair history.** Cross-aggregate: `pair_orders, pair_returns` для каждой `(buyer_id, seller_id)` пары. Captures «знаком ли уже этот покупатель с этим продавцом». Не пробовали. Подсчёт через DuckDB ~5 мин, фичи добавить — час.
- [ ] **Buyer TE с малым α.** Аналог `seller_te`, но для buyer_id (8M уникальных). С α≈20 — не пересглаживает, может дать сигнал.
- [ ] **Калибровка вероятностей.** isotonic / Platt на holdout. Помогает интерпретировать вероятности и устойчивее выбрать порог.
- [ ] **Анализ ошибок (для презентации финалистов!).** ТЗ требует «структуру прогноза по разрезам»: построить heatmap `precision/recall × city/category/seller_tenure`. Найти сегменты, где модель проседает, и обсудить в выводах.

### Не пробовали и не уверены, что выстрелит

- **Sequence модели (LSTM/Transformer на истории заказов покупателя)** — overhead огромный, прирост спекулятивный для табличной задачи.
- **Stacking с несколькими моделями разной природы (LR на TE-фичах + GBDT)** — обычно даёт меньше, чем blend двух хороших GBDT.
- **External данные** — нет, доступа нет.

---

## Окружение и подводные камни

### Окружение
- Python **3.14** (venv в корне репо: `Scripts/`, `Lib/`).
- Основные библиотеки: `polars==1.40`, `duckdb==1.5`, `lightgbm==4.6`, `catboost==1.2.10`, `pandas`, `numpy`, `scikit-learn`, `pyarrow`.
- ОС разработки: Windows 10/11 + PowerShell. На macOS/Linux всё должно работать, но кодировки и shell-синтаксис нужно адаптировать.

### Подводные камни (не наступай!)

- **`PYTHONIOENCODING=utf-8` обязательно** перед запуском с `print()`, иначе на cp1251 падает на кириллице.
- **`print()` в фоновой PowerShell-задаче ломает процесс** на cp1251. У всех скриптов вместо print — log() в файл (`artifacts/.../*.log`). Никогда не возвращай `print` обратно.
- **`sys.stdout.reconfigure()` тоже опасен в фоне**. Не вызывать.
- **polars streaming OOMит** на больших unique/sort. Тяжёлые агрегации — через DuckDB с `PRAGMA memory_limit='8GB'` и `temp_directory` в `artifacts/.../duckdb_tmp/`.
- **`pl.scan_csv(..., ignore_errors=True)`** — снимает 2049 broken rows в orders.csv (артефакт экспорта).
- **CatBoost не влезает в 16 ГБ на полный train.** Максимум — 22-25% сэмпл. Если у тебя 32+ ГБ RAM — попробуй на полном (вероятно +0.005-0.010 AUC).
- **Категориальные в LGBM** — всегда `pd.Categorical` с одинаковыми уровнями в train/holdout/test (см. `sync_categoricals` в скриптах). Иначе LGBM думает, что test содержит «новые» категории и даёт мусор.

---

## Что попробовали и не выстрелило

(чтобы не наступать на грабли повторно)

| Что | Результат | Почему |
|---|---|---|
| Rolling 7d/30d история отдельной фичей в LGBM C2 | 0 прироста | LGBM уже извлёк сигнал через `days_since_last_order` + all-time |
| Target encoding microcat/city для LGBM | 0 прироста | LGBM нативно работает с категориальными — TE избыточен |
| LGBM tuning sweep (deeper/regularized/scale_pos_weight) | -0.001-0.002 AUC | Текущие параметры (lr=0.05, leaves=63, min_data=200, is_unbalance=True) уже оптимальны |
| seller TE, dynamics diffs, price ratios, is_first_sale — все вместе (C4) | LGBM 0, CatBoost ~0 | Модели уже выжали потолок текущей информационной базы |
| Blend LGBM + CatBoost на 15% sample | +0.0003 AUC, +0.0004 F1 | CatBoost доминирует, LGBM добавляет немного диверсификации |

**Вывод:** на 15% сэмпле и текущих фичах потолок ≈ 0.805 AUC. Дальнейшие 0.005-0.010 ловятся ТОЛЬКО через бóльшие данные (что мы и делаем для финала) или через принципиально новые фичи (см. раздел «Куда копать»).

---

## Полезные ссылки

- ТЗ хакатона: `Задание на Хакатон ML4Pay от Avito.pdf` (в раздаче)
- FAQ ТЗ — ключевые правила оценки:
  - FAQ #2: «рассматривать каждый из товаров внутри заказа отдельно»
  - FAQ #3: «небольшая сезонность есть, но не сильная»
  - FAQ #7: «только 1 попытка сабмита, лидерборд после дедлайна»
- EDA отчёт: `artifacts/eda/` (графики и сводка)
- Логи всех тренировок: `artifacts/baseline_*/run.log`, `artifacts/lgbm_full/run.log`, `artifacts/catboost_22/run.log`
- Метрики по моделям: `artifacts/*/metrics.txt`

---

## Контакты команды

«ИИ в массы»: ваня бананов (`alexbessonov278@gmail.com`), \<остальные тиммейты\>

Если что-то поломалось при запуске — открой issue или напиши в наш чат. Логи в `artifacts/.../crash.log` обычно сразу показывают причину.

— _последнее обновление: 2026-05-08, в процессе финального прогона_
