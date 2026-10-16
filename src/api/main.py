import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import polars as pl
import torch
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from src.models.two_tower import TwoTowerModel

# Отключаем системный прокси для Qdrant
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"


# --- Схемы данных ответа (Pydantic) ---
class MovieRecommendation(BaseModel):
    rank: int
    movie_id: int
    title: str
    year: Optional[int]
    genres: list[str]
    confidence_score: float  # Скор от CatBoost
    retrieval_score: float   # Скор от Two-Tower


class RecommendationResponse(BaseModel):
    user_id: int
    total_latency_ms: float
    stage1_retrieval_ms: float
    stage2_ranking_ms: float
    candidates_retrieved: int
    recommendations: list[MovieRecommendation]


# --- Глобальное состояние сервиса в памяти ---
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация моделей и данных в оперативной памяти при старте сервера."""
    print("\n[STARTUP] Инициализация Two-Stage RecSys Engine...")
    t0 = time.perf_counter()

    # 1. Qdrant Client
    state["qdrant"] = QdrantClient(url="http://127.0.0.1:6333", timeout=10)

    # 2. Маппинги и Two-Tower
    mappings = torch.load("models/mappings.pt", weights_only=False)
    state["user2idx"] = mappings["user2idx"]
    state["movie2idx"] = mappings["movie2idx"]

    tt_model = TwoTowerModel(num_users=len(state["user2idx"]), num_movies=len(state["movie2idx"]))
    tt_model.load_state_dict(torch.load("models/two_tower.pt", weights_only=True))
    tt_model.eval()
    state["two_tower"] = tt_model

    # 3. CatBoost Ranker
    ranker = CatBoostClassifier()
    ranker.load_model("models/ranker.cbm")
    state["ranker"] = ranker

    # 4. In-Memory Feature Store (Polars -> быстрые словари)
    u_feat = pl.read_parquet("data/processed/user_features.parquet").with_columns(pl.col("user_id").cast(pl.Int32))
    m_feat = pl.read_parquet("data/processed/movie_features.parquet").with_columns(pl.col("movie_id").cast(pl.Int32))

    state["user_features"] = {row["user_id"]: row for row in u_feat.iter_rows(named=True)}
    state["movie_features"] = {row["movie_id"]: row for row in m_feat.iter_rows(named=True)}

    # 5. История просмотров пользователей (для исключения дублей)
    train = pl.read_parquet("data/processed/train.parquet")
    train_hist = train.group_by("user_id").agg(pl.col("movie_id").alias("seen"))
    state["user_seen"] = {row["user_id"]: set(row["seen"]) for row in train_hist.iter_rows(named=True)}

    # Самые популярные фильмы (Fallback на случай холодного старта)
    popular_movies = (
        train.group_by("movie_id")
        .len()
        .sort("len", descending=True)
        .limit(20)["movie_id"]
        .to_list()
    )
    state["popular_fallback"] = popular_movies

    startup_time = (time.perf_counter() - t0) * 1000
    print(f"[STARTUP] Система полностью готова за {startup_time:.2f} мс!\n")
    yield
    print("[SHUTDOWN] Остановка сервера...")


app = FastAPI(
    title="Real-Time RecSys Platform API",
    description="High-performance Two-Stage Recommendation Engine (Two-Tower + Qdrant + CatBoost)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def healthcheck():
    """Проверка жизнеспособности сервиса."""
    return {"status": "ok", "qdrant_connected": True}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: int,
    k: int = Query(default=10, ge=1, le=50, description="Количество рекомендаций"),
    genre: Optional[str] = Query(default=None, description="Опциональный фильтр по жанру"),
):
    """
    Главный эндпоинт сквозного Two-Stage инференса:
    1. Stage 1 (Two-Tower + Qdrant) -> отбор топ-100 кандидатов
    2. Stage 2 (CatBoost) -> тонкое ранжирование сотни кандидатов
    3. Выдача топ-K лучших фильмов
    """
    total_start = time.perf_counter()

    # Проверка на холодного пользователя
    user2idx = state["user2idx"]
    if user_id not in user2idx or user_id not in state["user_features"]:
        raise HTTPException(
            status_code=404,
            detail=f"User #{user_id} не найден. Используйте ID от 1 до 6040 или эндпоинт холодного старта.",
        )

    # ==========================================
    # СТАДИЯ 1: Candidate Retrieval (Qdrant)
    # ==========================================
    t1_start = time.perf_counter()
    uidx = torch.tensor([user2idx[user_id]])
    with torch.no_grad():
        user_vector = state["two_tower"].user_tower(uidx).squeeze(0).tolist()

    seen_movies = list(state["user_seen"].get(user_id, set()))

    must_not = [models.HasIdCondition(has_id=seen_movies)] if seen_movies else None
    must = [models.FieldCondition(key="genres", match=models.MatchValue(value=genre))] if genre else None

    query_filter = models.Filter(must=must, must_not=must_not)

    # Запрашиваем 100 кандидатов из Qdrant
    qdrant_results = state["qdrant"].query_points(
        collection_name="movies",
        query=user_vector,
        query_filter=query_filter,
        limit=100,
        with_payload=True,
    )
    t1_elapsed = (time.perf_counter() - t1_start) * 1000

    if not qdrant_results.points:
        raise HTTPException(status_code=404, detail="Кандидаты по заданным фильтрам не найдены.")

    # ==========================================
    # СТАДИЯ 2: Heavy Ranking (CatBoost)
    # ==========================================
    t2_start = time.perf_counter()
    u_f = state["user_features"][user_id]
    m_feat_dict = state["movie_features"]

    candidate_rows = []
    for point in qdrant_results.points:
        mid = point.id
        m_f = m_feat_dict.get(mid)
        if not m_f:
            continue

        candidate_rows.append({
            "gender": u_f["gender"],
            "age": u_f["age"],
            "occupation": u_f["occupation"],
            "user_rating_count": u_f["user_rating_count"],
            "user_avg_rating": u_f["user_avg_rating"],
            "release_year": m_f["release_year"],
            "movie_rating_count": m_f["movie_rating_count"],
            "movie_avg_rating": m_f["movie_avg_rating"],
            "movie_std_rating": m_f["movie_std_rating"],
            "genres": m_f["genres"],
            "rating_diff": m_f["movie_avg_rating"] - u_f["user_avg_rating"],
            "two_tower_score": point.score,
            "movie_id": mid,
            "title": point.payload.get("title", f"Movie {mid}"),
            "year": point.payload.get("year"),
            "genres_list": point.payload.get("genres", []),
        })

    cand_df = pl.DataFrame(candidate_rows)

    feature_cols = [
        "gender", "age", "occupation", "user_rating_count", "user_avg_rating",
        "release_year", "movie_rating_count", "movie_avg_rating", "movie_std_rating",
        "genres", "rating_diff", "two_tower_score"
    ]

    X_cand = cand_df.select(feature_cols).to_pandas()
    X_cand["gender"] = X_cand["gender"].fillna("Unknown")
    X_cand["genres"] = X_cand["genres"].fillna("Unknown")

    # Быстрый инференс CatBoost по 100 кандидатам
    pool = Pool(X_cand, cat_features=["gender", "genres"])
    probs = state["ranker"].predict_proba(pool)[:, 1]
    t2_elapsed = (time.perf_counter() - t2_start) * 1000

    cand_df = cand_df.with_columns(pl.Series("catboost_prob", probs))
    top_k_df = cand_df.sort("catboost_prob", descending=True).head(k)

    # ==========================================
    # Формирование ответа
    # ==========================================
    total_elapsed = (time.perf_counter() - total_start) * 1000

    recommendations = []
    for rank_idx, row in enumerate(top_k_df.iter_rows(named=True), start=1):
        recommendations.append(
            MovieRecommendation(
                rank=rank_idx,
                movie_id=row["movie_id"],
                title=row["title"],
                year=row["year"],
                genres=row["genres_list"],
                confidence_score=round(float(row["catboost_prob"]), 4),
                retrieval_score=round(float(row["two_tower_score"]), 4),
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        total_latency_ms=round(total_elapsed, 2),
        stage1_retrieval_ms=round(t1_elapsed, 2),
        stage2_ranking_ms=round(t2_elapsed, 2),
        candidates_retrieved=len(qdrant_results.points),
        recommendations=recommendations,
    )