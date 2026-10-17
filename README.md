# ⚡ PulseRec Engine // Two-Stage Real-Time Neural Recommendation Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x%20CUDA%2012.4-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CatBoost-YetiRank%20%2F%20GBDT-FFCC00?style=for-the-badge&logo=yandex&logoColor=black" />
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB%20HNSW-DC2626?style=for-the-badge&logo=qdrant&logoColor=white" />
  <img src="https://img.shields.io/badge/Redis-Feature%20Store%20%26%20Streams-DC382D?style=for-the-badge&logo=redis&logoColor=white" />
  <img src="https://img.shields.io/badge/Polars-Lightning%20Fast%20DataFrames-CD792C?style=for-the-badge&logo=polars&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Production%20Async-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Three.js-WebGL%203D%20Galaxy-000000?style=for-the-badge&logo=three.js&logoColor=white" />
  <img src="https://img.shields.io/badge/Package%20Manager-Astral%20uv-DE5FE9?style=for-the-badge&logo=uv&logoColor=white" />
</p>

Промышленная двухстадийная рекомендательная система (**Two-Stage Recommender System**) полного цикла с динамической адаптацией под сессию пользователя в реальном времени, потоковым Feature Store в памяти, гарантированным разнообразием выдачи (MMR) и exploration-бандитами.

Платформа масштабирована от **MovieLens 1M** до **MovieLens 25M** (10+ млн взаимодействий с обработкой прямо в GPU VRAM) и снабжена интерактивным WebGL-фронтендом для исследования скрытого пространства эмбеддингов.

---

## 🏛 Архитектура Системы (High-Level Architecture)

```
[Пользователь кликает / лайкает фильм]
                 │
                 ▼
 ┌───────────────────────────────┐
 │   FastAPI Gateway (/feedback) │
 └───────────────┬───────────────┘
                 │
                 ├──► [Redis Streams] ──► XADD 'events:feedback' (Async Analytics)
                 └──► [Redis Online Feature Store] ──► LPUSH session:{uid} (TTL 1h)
                                                              │
[Запрос рекомендаций: GET /recommend/{uid}]                   │
                 │                                            │
                 ▼                                            │
 ┌───────────────────────────────┐                            │
 │  Real-Time Session Blending   │◄───────────────────────────┘
 │  u_live = 0.65·u0 + 0.35·mean │ (Адаптация вектора без переобучения за < 0.5 мс)
 └───────────────┬───────────────┘
                 │
                 ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ СТАДИЯ 1: Кандидатная выборка (Candidate Retrieval)          │
 │ Qdrant Vector DB (HNSW Index, Cosine Similarity, Top-100)   │
 │ Latency: 2–4 мс | Фильтрация просмотренного (History Mask)  │
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
                 │ Топ-10 рекомендаций
                 ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Интерактивный UI (React 19 + Three.js 3D Latent Galaxy)     │
 │ Телеметрия: Prometheus (/metrics) + Grafana Dashboard       │
 └─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Математический Аппарат и Модели

### 1. Stage 1: Двухбашенная нейросеть (Two-Tower DSSM)
Пользователи и фильмы проецируются в общее 64-мерное гиперсферическое пространство:

$$\mathbf{u} = \text{Tower}_{\text{user}}(\text{uid}), \quad \mathbf{v} = \text{Tower}_{\text{item}}(\text{mid}), \quad \|\mathbf{u}\|_2 = 1, \ \|\mathbf{v}\|_2 = 1$$

* **Архитектура башни**: `Embedding(dim=64)` $\rightarrow$ `Linear(64, 128)` $\rightarrow$ `LayerNorm(128)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(128, 64)` $\rightarrow$ `$L_2$ Normalization`.
* **Функция потерь (Symmetric InfoNCE Loss с In-Batch Negatives)**:
  Для батча размера $B$ на диагонали матрицы скалярных произведений находятся истинные пары взаимодействия, а остальные $B-1$ элементов строки выступают жесткими негативными примерами:
  $$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp\left(\frac{\mathbf{u}_i \cdot \mathbf{v}_i}{\tau}\right)}{\sum_{j=1}^{B} \exp\left(\frac{\mathbf{u}_i \cdot \mathbf{v}_j}{\tau}\right)}, \quad \tau = 0.07$$

### 2. Real-Time Сессионный Blending (Online Session Adaptation)
Вместо дорогостоящего дообучения модели на лету, сессионные предпочтения пользователя подмешиваются непосредственно в скрытом пространстве:

$$\mathbf{u}_{\text{session}} = \text{Normalize}\left(0.65 \cdot \mathbf{u}_{\text{base}} + 0.35 \cdot \frac{1}{|S|} \sum_{m \in S} \mathbf{v}_m\right)$$

где $S$ — последние лайки пользователя из кольцевого буфера Redis (`LTRIM 0 9`, TTL 3600с).

### 3. Stage 2: Градиентный бустинг (CatBoost Re-Ranker)
Ранжирует топ-100 кандидатов первой стадии на основе 12 плотных и категориальных признаков:
* **Признаки взаимодействия**: `two_tower_score` (косинус первой стадии), `rating_diff = movie_avg - user_avg`.
* **Признаки пользователя**: `gender`, `age`, `occupation`, `user_rating_count`, `user_avg_rating`.
* **Признаки контента**: `release_year`, `movie_rating_count`, `movie_avg_rating`, `movie_std_rating`, `genres`.

### 4. Контроль разнообразия (Maximal Marginal Relevance, MMR)
Предотвращает проблему «пузыря фильтров» (Filter Bubble), когда в топе оказываются фильмы только одного жанра/франшизы:

$$\text{MMR} = \arg\max_{c \in C \setminus S} \left[ (1 - \lambda) \cdot P_{\text{CatBoost}}(c) - \lambda \cdot \max_{s \in S} \text{Jaccard}(G_c, G_s) \right]$$

где $G$ — множество жанров фильма, $\lambda \in [0, 1]$ регулируется ползунком на фронтенде в реальном времени.

### 5. Multi-Armed Bandit Exploration (Разведка новинок)
Алгоритм $\epsilon$-Greedy с оценкой потенциальной неопределенности отбирает фильмы с высоким скором нейросети, но малым числом исторических оценок:

$$\text{BanditPotential}(i) = \frac{\text{TwoTowerScore}(i)}{(\text{RatingCount}_i + 1)^{0.3}}$$

---

## 📊 Результаты и Бенчмарки (Offline & Online Evaluation)

### Сравнение этапов пайплайна (MovieLens 1M, честный Temporal Split 80/20)
Тестирование проводилось с обязательным **маскированием всех фильмов из истории Train** (зануление скоров до $-\infty$):

| Модель / Этап | HitRate@10 | HitRate@100 | NDCG@10 | Примечание |
| :--- | :---: | :---: | :---: | :--- |
| **Popularity Baseline** | 0.2463 | — | 0.0396 | Неперсонализированный топ популярных |
| **Stage 1 (Two-Tower alone)** | 0.2026 | **0.5841** | 0.0309 | Отбирает 100 семантических кандидатов |
| **Full Two-Stage (Two-Tower + CatBoost)** | **0.2580** | — | **0.0435** | **+27.3% по HitRate@10** относительно Stage 1 |

### Метрики CatBoost Ranker (Валидационная выборка)
Обучение с ранней остановкой (`early_stopping_rounds=40`) по метрике `AUC`:

* **Validation AUC**: `0.9265`
* **Validation Logloss**: `0.3315` (падение с базового `0.6306`)

```
=============================================
ВАЖНОСТЬ ПРИЗНАКОВ (FEATURE IMPORTANCE):
=============================================
two_tower_score      |  34.12% █████████████████
rating_diff          |  18.45% █████████
movie_avg_rating     |  14.20% ███████
movie_rating_count   |   9.80% ████
genres               |   7.45% ███
user_avg_rating      |   5.12% ██
release_year         |   4.10% ██
user_rating_count    |   3.15% █
occupation           |   1.80% 
age                  |   1.10% 
gender               |   0.71% 
=============================================
```

### Нагрузочное тестирование (Locust Stress Test)
Тест проводился при одновременной симуляции **50 активных пользователей**, непрерывно запрашивающих рекомендации с инференсом CatBoost и векторным поиском:

* **Sustained Throughput**: `~47 RPS` (запросов в секунду)
* **Failures**: `0.0%` (0 ошибок на всей дистанции)
* **p50 Response Time**: `~650–700 ms` (на CPU под 100% утилизацией)
* **p95 Response Time**: `~1150–1200 ms`
* **Single Request Latency (без нагрузки)**: `~8–15 ms` (Qdrant: 2-3ms, CatBoost: 5-8ms).

---

## 🚀 Турбо-пайплайн MovieLens 25M (Streaming & In-GPU Execution)

Для масштабирования на датасет **MovieLens 25M (25 000 000 строк)** реализованы оптимизации корпоративного уровня:
1. **Lazy Streaming CSV в Parquet**: обработка без OOM через `polars.scan_csv()` и потоковый `sink_parquet()` со сжатием `zstd`.
2. **Global Temporal Cutoff**: разделение по 80-му квантилю глобального timestamp для сохранения хронологии.
3. **10M Rows In-VRAM Training**: 10 млн плотных `int64` индексов обучающих пар загружаются **целиком в память GPU** (занимает всего ~160 МБ VRAM).
4. **Zero-CPU Batching**: Перетасовка эпохи происходит на видеокарте (`torch.randperm(device="cuda")`), батчи нарезаются без участия шины PCIe и CPU. Скорость обучения — **более 1 000 000 взаимодействий в секунду**!

---

## 🎨 Интерактивный UI & 3D WebGL Galaxy

Фронтенд построен на **React 19**, **Vite** и **Three.js**:
* **3D Two-Tower Latent Space**: трехмерная интерактивная галактика из 2000 фоновых звезд каталога и планет Top-10 выдачи. В центре сферы визуализирован динамический вектор пользователя, соединенный лучами с кандидатами.
* **Official Posters & Smart Clean-up**: автоматический каскадный парсер обложек (OMDb IMDb API $\rightarrow$ Wikipedia API $\rightarrow$ Wikipedia REST Summary) с очисткой внутренних меток датасета (`(ID4)`, альтернативных названий и артиклей).
* **Прямая интеграция с Netflix**: мгновенный переход на просмотр найденного фильма по клику на карточку или постер.
* **ML Inspector**: выдвижная панель с декомпозицией скоров (вклад косинуса первой стадии vs уверенность CatBoost).

---

## 🛠 Технологический Стек (Stack & Infrastructure)

| Область | Технологии |
| :--- | :--- |
| **Язык & Менеджер пакетов** | Python 3.12, Astral `uv` (детерминированный `uv.lock`) |
| **Deep Learning (Stage 1)** | PyTorch 2.x (CUDA 12.4), LayerNorm, Symmetric InfoNCE Loss |
| **Vector Database** | Qdrant (HNSW Index, Cosine metric, Payload filtering, gRPC/REST) |
| **Ranking ML (Stage 2)** | CatBoostClassifier / YetiRank, Early Stopping, AUC Optimization |
| **Feature Store & State** | Redis (LPUSH/LTRIM sessions, TTL, XADD Event Stream), Polars Parquet |
| **API & Микросервисы** | FastAPI, Uvicorn, Pydantic V2, Docker Multi-Stage Build |
| **Мониторинг** | Prometheus (`prometheus-fastapi-instrumentator`), Grafana, Locust |
| **Frontend & 3D** | React 19, Three.js, Lucide Icons, Tailwind CSS, Framer Motion |

---

## 🚦 Быстрый Старт (Quickstart)

### 1. Клонирование и запуск инфраструктуры
Для работы требуются установленный Docker и Docker Compose:

```bash
git clone https://github.com/NikiChaZi/recsys-two-stage-engine.git
cd recsys-two-stage-engine

# Запуск Qdrant, Redis, Prometheus и Grafana
docker compose up -d
```

Инфраструктура доступна по адресам:
* **Qdrant Web UI**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
* **Prometheus**: [http://localhost:9090](http://localhost:9090)
* **Grafana**: [http://localhost:3000](http://localhost:3000) (логин: `admin`, пароль: `admin`)

---

### 2. Установка зависимостей через Astral `uv`
Проект использует сверхбыстрый пакетный менеджер `uv`:

```bash
# Установка uv (если еще не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Синхронизация окружения (виртуальное окружение создастся автоматически)
uv sync
```

---

### 3. Подготовка данных и обучение моделей

```bash
# 1. Скачивание MovieLens 1M и темпоральный сплит 80/20
uv run python src/data/download.py

# 2. Извлечение признаков пользователей и фильмов в Polars Parquet
uv run python src/features/build_features.py

# 3. Обучение Stage 1: Two-Tower модели (PyTorch InfoNCE)
uv run python src/models/train.py

# 4. Обучение Stage 2: CatBoost ранжировщика
uv run python src/models/train_catboost.py

# (Опционально) Запуск полной оффлайн-валидации
uv run python src/evaluation/evaluate_full_pipeline.py
```

---

### 4. Запуск бэкенда и фронтенда

Запуск API сервера:
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Запуск клиентского приложения:
```bash
cd frontend
npm install
npm run dev
```

Откройте в браузере: **[http://localhost:5173](http://localhost:5173)**.

---

## 📡 Спецификация REST API

### `GET /recommend/{user_id}`
Получить персональный топ-K рекомендаций с учетом сессии в Redis.

**Параметры запроса:**
* `k` (int, default: 10): размер итоговой выдачи.
* `genre` (str, optional): фильтрация по жанру на этапе векторного поиска в Qdrant.
* `diversity` (float, default: 0.0, max: 1.0): коэффициент MMR-штрафа за схожесть жанров.
* `exploration_rate` (float, default: 0.1, max: 0.5): доля подмешивания бандита (Uncertainty Exploration).

**Пример ответа:**
```json
{
  "user_id": 42,
  "total_latency_ms": 11.42,
  "stage1_retrieval_ms": 3.12,
  "stage2_ranking_ms": 6.84,
  "candidates_retrieved": 100,
  "diversity_applied": 0.4,
  "exploration_rate": 0.1,
  "exploration_items_count": 1,
  "session_source": "Redis Online Feature Store",
  "recommendations": [
    {
      "rank": 1,
      "movie_id": 296,
      "title": "Pulp Fiction",
      "year": 1994,
      "genres": ["Crime", "Drama"],
      "confidence_score": 0.9842,
      "retrieval_score": 0.8912,
      "is_exploration": false
    }
  ]
}
```

### `POST /feedback`
Записать действие пользователя (лайк/просмотр) для мгновенной адаптации сессионного вектора.

```json
{
  "user_id": 42,
  "movie_id": 1193,
  "rating": 5.0
}
```

---

## 📈 Нагрузочное тестирование (Locust)

Для запуска стресс-тестов:
```bash
uv run locust -f benchmarks/locustfile.py --host http://localhost:8000
```
Интерфейс тестирования доступен по адресу: [http://localhost:8089](http://localhost:8089).

---

## 👨‍💻 Автор

**Никита (NikiChaZi)**  
* GitHub: [@NikiChaZi](https://github.com/NikiChaZi)  
* Проект: [recsys-two-stage-engine](https://github.com/NikiChaZi/recsys-two-stage-engine)