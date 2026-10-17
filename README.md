# ⚡ PulseRec Engine // Two-Stage Real-Time Neural Recommendation Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x%20CUDA%2012.4-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CatBoost-GBDT%20Ranker-FFCC00?style=for-the-badge&logo=yandex&logoColor=black" />
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB%20HNSW-DC2626?style=for-the-badge&logo=qdrant&logoColor=white" />
  <img src="https://img.shields.io/badge/Redis-Feature%20Store%20%26%20Streams-DC382D?style=for-the-badge&logo=redis&logoColor=white" />
  <img src="https://img.shields.io/badge/Polars-LazyFrames%20ETL-CD792C?style=for-the-badge&logo=polars&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Async%20Serving-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React%2019-Three.js%203D%20Galaxy-000000?style=for-the-badge&logo=react&logoColor=white" />
  <img src="https://img.shields.io/badge/Package%20Manager-Astral%20uv-DE5FE9?style=for-the-badge&logo=uv&logoColor=white" />
</p>

Промышленная двухстадийная рекомендательная система (**Two-Stage Recommender System**) полного цикла: от векторного поиска кандидатов в **Qdrant (HNSW)** до тяжелого переранжирования на **CatBoost**, с динамической адаптацией под сессию в **Redis Streams**, алгоритмом разнообразия (MMR), exploration-бандитами и 3D WebGL-визуализацией скрытого пространства эмбеддингов.

Платформа масштабирована от **MovieLens 1M** до **MovieLens 25M** (потоковая обработка 25 млн строк на Polars LazyFrames и обучение в VRAM GPU на 80+ it/s).

---

## 🏛 Архитектура Системы (Architecture Flow)

```text
[Клиент: клик / лайк фильма]
                 │
                 ▼
 ┌───────────────────────────────┐
 │   FastAPI Gateway (/feedback) │
 └───────────────┬───────────────┘
                 ├──► [Redis Streams] ──► XADD 'events:feedback' (Асинхронная аналитика)
                 └──► [Redis Online Feature Store] ──► LPUSH session:{uid} (TTL 1h)
                                                              │
[Запрос ленты: GET /recommend/{uid}]                          │
                 │                                            │
                 ▼                                            │
 ┌───────────────────────────────┐                            │
 │  Real-Time Session Blending   │◄───────────────────────────┘
 │  u_live = 0.65·u0 + 0.35·mean │ (Адаптация вектора без переобучения за < 0.2 мс)
 └───────────────┬───────────────┘
                 │
                 ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ СТАДИЯ 1: Кандидатная выборка (Candidate Retrieval)          │
 │ Qdrant Vector DB (24,659 фильмов, HNSW Index, Cosine)       │
 │ Задержка: 14–25 мс | Фильтрация просмотренного (History)    │
 └───────────────┬─────────────────────────────────────────────┘
                 │ 100 кандидатов
                 ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ СТАДИЯ 2: Тяжелое переранжирование (Heavy Re-Ranking)       │
 │ In-Memory Polars Feature Store + CatBoost GBDT (AUC: 0.926) │
 │ 12 признаков: Two-Tower Score, Rating Diff, User/Item Stats │
 └───────────────┬─────────────────────────────────────────────┘
                 │ Ранжированный скор вероятности P(like)
                 ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ ПОСТ-ПРОЦЕССИНГ & БИЗНЕС-ЛОГИКА                             │
 │ 1. MMR Diversity: штраф за пересечение жанров (Jaccard)     │
 │ 2. ε-Greedy Bandit: подмешивание "скрытых жемчужин"         │
 └───────────────┬─────────────────────────────────────────────┘
                 │ Топ-10 рекомендаций (< 35 мс)
                 ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Интерактивный UI (React + Three.js WebGL Galaxy + Posters)  │
 │ MLOps Наблюдаемость: Prometheus (/metrics) + Grafana (:3000) │
 └─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Математический Аппарат

### 1. Stage 1: Двухбашенная нейросеть (Two-Tower DSSM)
Пользователи и фильмы проецируются на единичную 64-мерную гиперсферу:

$$
\mathbf{u} = \text{Tower}_{\text{user}}(\text{uid}), \quad \mathbf{v} = \text{Tower}_{\text{item}}(\text{mid}), \quad \|\mathbf{u}\|_2 = 1, \quad \|\mathbf{v}\|_2 = 1
$$

* **Архитектура**: `Embedding(64)` $\rightarrow$ `Linear(64, 128)` $\rightarrow$ `LayerNorm(128)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(128, 64)` $\rightarrow$ $L_2$ Normalization.
* **Функция потерь (Symmetric InfoNCE Loss с In-Batch Negatives)**:

$$
\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{\text{user}\to\text{item}} + \mathcal{L}_{\text{item}\to\text{user}} \right), \quad \tau = 0.07
$$

### 2. Stage 2: Градиентный бустинг (Pointwise CatBoost Ranker)
Сотня отобранных кандидатов ранжируется ансамблем решающих деревьев CatBoost по 12 признакам:
* **Кросс-признаки**: `two_tower_score` (косинус первой стадии), `rating_diff = movie_avg - user_avg`.
* **Признаки пользователя**: `gender`, `age`, `occupation`, `user_rating_count`, `user_avg_rating`.
* **Признаки контента**: `release_year`, `movie_rating_count`, `movie_avg_rating`, `movie_std_rating`, `genres`.

### 3. Бизнес-логика: MMR Diversity & Multi-Armed Bandit
* **Maximal Marginal Relevance (MMR)**: штрафует схожесть жанров по метрике Жаккара для предотвращения информационного пузыря:

$$
\text{MMR} = \arg\max_{c \in C \setminus S} \left[ (1 - \lambda) \cdot P_{\text{CatBoost}}(c) - \lambda \cdot \max_{s \in S} \text{Jaccard}(G_c, G_s) \right]
$$

* **$\epsilon$-Greedy Bandit (Uncertainty Exploration)**: подмешивает фильмы с высоким нейросетевым потенциалом, но малым числом оценок:

$$
\text{ExplorationScore}(i) = \frac{\text{TwoTowerScore}(i)}{(\text{RatingCount}_i + 1)^{0.3}}
$$

---

## 📊 Результаты и Бенчмарки (Фактические метрики)

### Сравнение этапов пайплайна (Честный Temporal Split 80/20)
Тестирование проводилось с обязательным маскированием просмотренных фильмов из Train ($scores = -\infty$):

| Модель / Конфигурация | HitRate@10 | HitRate@100 | NDCG@10 | Прирост к Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.2463 | — | 0.0396 | — |
| **Stage 1 (Two-Tower Retrieval)** | 0.2026 | **0.7648** | 0.0309 | Шорт-лист кандидатов |
| **Full Two-Stage (Two-Tower + CatBoost)** | **0.4313** | — | **0.0806** | **+75.1% HR / +103.5% NDCG** |

> **Ключевой вывод**: Первая стадия сужает каталог в 35 раз с сохранением **76.5% релевантных объектов**, а CatBoost на второй стадии более чем **удваивает качество сортировки NDCG@10 (с 0.0396 до 0.0806)**.

### Важность признаков CatBoost (Validation AUC: 0.9265)
```text
movie_rating_count   |  34.63% █████████████████
two_tower_score      |  29.22% ██████████████
user_rating_count    |  11.30% █████
rating_diff          |   9.66% ████
movie_avg_rating     |   8.88% ████
user_avg_rating      |   1.83% █
age                  |   1.40% 
occupation           |   1.32% 
gender               |   0.55% 
release_year         |   0.52% 
```

### Нагрузочное тестирование (Locust, 50 concurrent users)
* **Пропускная способность**: `47.8 RPS` (на 1 рабочем процессе)
* **Ошибки**: `0.0%` (0 сбоев на всей дистанции)
* **Single Request Latency E2E**: `~25–35 ms` (Qdrant: ~14 ms, CatBoost: ~19 ms).
* **Оптимизация ONNX Runtime (CPU)**: инференс User Tower ускорен с `0.198 ms` (PyTorch) до **`0.036 ms` (5.43x speedup)**.

---

## 🛠 Структура Репозитория

```text
recsys-two-stage-engine/
├── configs/
│   └── prometheus.yml          <- Конфигурация сбора метрик Prometheus
├── data/
│   ├── raw/                    <- Исходные датасеты (ml-1m, ml-25m)
│   └── processed/              <- Parquet-файлы с признаками и сплитами
├── benchmarks/
│   ├── locustfile.py           <- Сценарий нагрузочного тестирования
│   └── locust_results.png      <- Графики RPS и перцентилей задержки
├── frontend/                   <- React 19 + Three.js + Tailwind UI
│   ├── src/components/Galaxy3D.jsx <- WebGL 3D визуализатор эмбеддингов
│   └── src/App.jsx             <- Кинематографичный клиент Criterion Style
├── models/                     <- Артефакты весов (PT, ONNX, CBM)
├── src/
│   ├── api/
│   │   └── main.py             <- FastAPI сервис с эндпоинтами и lifespan
│   ├── data/
│   │   ├── download.py         <- Загрузчик и temporal split (ML-1M)
│   │   ├── prepare_25m.py      <- Lazy Streaming пайплайн (ML-25M)
│   │   ├── dataset.py          <- PyTorch Dataset и индексация
│   │   └── features.py         <- Витрина признаков на Polars
│   ├── models/
│   │   ├── two_tower.py        <- Архитектура башен с LayerNorm
│   │   └── export_onnx_25m.py  <- Экспорт и бенчмарк ONNX Runtime
│   ├── training/
│   │   ├── train_retrieval.py  <- Обучение Two-Tower на 1M
│   │   ├── train_25m.py        <- Обучение Two-Tower в VRAM на 10M строках
│   │   └── train_ranker.py     <- Обучение CatBoost GBDT
│   ├── evaluation/
│   │   ├── metrics.py          <- Реализация HitRate@K и NDCG@K
│   │   ├── baseline.py         <- Бейзлайн популярности
│   │   └── evaluate_two_stage.py <- Полный замер двух стадий
│   └── services/
│       ├── redis_store.py      <- Redis Online Feature Store & Streams
│       ├── index_qdrant.py     <- Индексация 1M каталога в Qdrant
│       └── index_qdrant_25m.py <- Индексация 24.6k фильмов в Qdrant
├── docker-compose.yml          <- Qdrant, Redis, Prometheus, Grafana
├── pyproject.toml              <- Декларация зависимостей uv
└── main.py                     <- Корневая точка входа Uvicorn
```

---

## 🚦 Воспроизводимый запуск (Step-by-Step Quickstart)

### 1. Запуск инфраструктуры в Docker
```bash
# Поднимаем Qdrant (6333), Redis (6379), Prometheus (9090) и Grafana (3000)
docker compose up -d
```

### 2. Установка зависимостей через Astral `uv`
```bash
# Синхронизация зависимостей
uv sync
```

### 3. Воспроизведение полного ML-пайплайна
Все скрипты запускаются строго как модули из корня репозитория:

```bash
# Шаг 1: Подготовка данных и честный Temporal Split
uv run python -m src.data.download

# Шаг 2: Генерация признаков пользователей и фильмов в Polars
uv run python -m src.data.features

# Шаг 3: Обучение Stage 1 Two-Tower модели
uv run python -m src.training.train_retrieval

# Шаг 4: Заливка эмбеддингов в векторную базу Qdrant
uv run python -m src.services.index_qdrant

# Шаг 5: Обучение Stage 2 CatBoost ранжировщика
uv run python -m src.training.train_ranker

# Шаг 6: Запуск полного сквозного бенчмарка метрик
uv run python -m src.evaluation.evaluate_two_stage
```

### 4. Запуск бэкенда и мониторинга
```bash
# Запуск FastAPI сервиса
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Swagger UI документация**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Grafana ЦУП**: [http://localhost:3000](http://localhost:3000) (admin / admin)
* **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### 5. Запуск интерактивного фронтенда
```bash
cd frontend
npm install
npm run dev
```
Откройте в браузере: **[http://localhost:5173](http://localhost:5173)**.

---

## 📈 Нагрузочное тестирование
```bash
uv run locust -f benchmarks/locustfile.py --host http://localhost:8000
```
Дашборд доступен на [http://localhost:8089](http://localhost:8089).

---

## 👨‍💻 Автор
**Никита (NikiChaZi)**  
* GitHub: [@NikiChaZi](https://github.com/NikiChaZi)  
* Проект: [recsys-two-stage-engine](https://github.com/NikiChaZi/recsys-two-stage-engine)