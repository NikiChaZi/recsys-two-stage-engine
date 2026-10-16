from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import os
import random
import time
from contextlib import asynccontextmanager
from typing import Optional

import polars as pl
import torch
import torch.nn.functional as F
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from src.models.two_tower import TwoTowerModel
from src.services.redis_store import RedisSessionStore

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"


# --- Pydantic Схемы ---
class MovieRecommendation(BaseModel):
    rank: int
    movie_id: int
    title: str
    year: Optional[int]
    genres: list[str]
    confidence_score: float
    retrieval_score: float
    is_exploration: bool = False


class RecommendationResponse(BaseModel):
    user_id: int
    total_latency_ms: float
    stage1_retrieval_ms: float
    stage2_ranking_ms: float
    candidates_retrieved: int
    diversity_applied: float
    exploration_rate: float
    exploration_items_count: int
    session_source: str
    recommendations: list[MovieRecommendation]


class FeedbackRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float = Field(ge=1.0, le=5.0)


class FeedbackResponse(BaseModel):
    status: str
    message: str
    event_stream: str


state = {}


# --- Алгоритмы разнообразия и бандитов ---
def genre_jaccard_similarity(genres_a: list[str], genres_b: list[str]) -> float:
    set_a, set_b = set(genres_a), set(genres_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def apply_mmr_diversity(candidate_rows: list[dict], top_k: int, lambda_div: float) -> list[dict]:
    if lambda_div <= 0.0 or len(candidate_rows) <= top_k:
        return candidate_rows[:top_k]

    selected = []
    candidates = candidate_rows.copy()
    first = max(candidates, key=lambda x: x["catboost_prob"])
    selected.append(first)
    candidates.remove(first)

    while len(selected) < top_k and candidates:
        best_item = None
        best_score = -float("inf")

        for cand in candidates:
            rel = cand["catboost_prob"]
            max_sim = max(
                genre_jaccard_similarity(cand["genres_list"], sel["genres_list"])
                for sel in selected
            )
            mmr_score = (1.0 - lambda_div) * rel - lambda_div * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_item = cand

        selected.append(best_item)
        candidates.remove(best_item)

    return selected


def apply_bandit_exploration(
    candidate_rows: list[dict],
    top_k: int,
    exploration_rate: float = 0.1,
) -> tuple[list[dict], int]:
    """
    Epsilon-Greedy Multi-Armed Bandit:
    Подмешивает скрытые жемчужины с высоким потенциалом (Uncertainty Exploration).
    """
    if exploration_rate <= 0.0 or len(candidate_rows) <= top_k:
        return candidate_rows[:top_k], 0

    for cand in candidate_rows:
        count = cand.get("movie_rating_count", 0)
        tt_score = max(0.01, float(cand.get("two_tower_score", 0.5)))
        cand["bandit_potential"] = tt_score / ((count + 1) ** 0.3)

    num_explore = max(1, int(top_k * exploration_rate))
    num_exploit = top_k - num_explore

    exploit_items = candidate_rows[:num_exploit]
    exploit_ids = {x["movie_id"] for x in exploit_items}

    remaining = [x for x in candidate_rows[num_exploit:] if x["movie_id"] not in exploit_ids]
    remaining.sort(key=lambda x: x["bandit_potential"], reverse=True)
    explore_items = remaining[:num_explore]

    for item in explore_items:
        item["is_bandit_explore"] = True

    combined = exploit_items.copy()
    for exp_item in explore_items:
        insert_pos = random.randint(min(3, len(combined)), len(combined))
        combined.insert(insert_pos, exp_item)

    return combined[:top_k], len(explore_items)


# --- Жизненный цикл сервиса ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[STARTUP] Инициализация Two-Stage RecSys Engine с Redis...")
    t0 = time.perf_counter()

    qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    state["qdrant"] = QdrantClient(url=qdrant_url, timeout=10)

    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    state["redis_store"] = RedisSessionStore(redis_url=redis_url)

    mappings = torch.load("models/mappings.pt", weights_only=False)
    state["user2idx"] = mappings["user2idx"]
    state["movie2idx"] = mappings["movie2idx"]

    tt_model = TwoTowerModel(num_users=len(state["user2idx"]), num_movies=len(state["movie2idx"]))
    tt_model.load_state_dict(torch.load("models/two_tower.pt", weights_only=True))
    tt_model.eval()
    state["two_tower"] = tt_model

    ranker = CatBoostClassifier()
    ranker.load_model("models/ranker.cbm")
    state["ranker"] = ranker

    u_feat = pl.read_parquet("data/processed/user_features.parquet").with_columns(pl.col("user_id").cast(pl.Int32))
    m_feat = pl.read_parquet("data/processed/movie_features.parquet").with_columns(pl.col("movie_id").cast(pl.Int32))

    state["user_features"] = {row["user_id"]: row for row in u_feat.iter_rows(named=True)}
    state["movie_features"] = {row["movie_id"]: row for row in m_feat.iter_rows(named=True)}

    train = pl.read_parquet("data/processed/train.parquet")
    train_hist = train.group_by("user_id").agg(pl.col("movie_id").alias("seen"))
    state["user_seen"] = {row["user_id"]: set(row["seen"]) for row in train_hist.iter_rows(named=True)}

    startup_time = (time.perf_counter() - t0) * 1000
    print(f"[STARTUP] Система готова за {startup_time:.2f} мс!\n")
    yield
    print("[SHUTDOWN] Остановка сервера...")


app = FastAPI(
    title="Real-Time RecSys Platform API",
    description="Two-Stage RecSys Engine with Redis Feature Store, MMR & Multi-Armed Bandits",
    version="1.3.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)

@app.get("/health")
def healthcheck():
    redis_alive = state["redis_store"].ping()
    return {"status": "ok", "qdrant_connected": True, "redis_connected": redis_alive}


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackRequest):
    user_id = feedback.user_id
    movie_id = feedback.movie_id

    if user_id not in state["user_seen"]:
        state["user_seen"][user_id] = set()
    state["user_seen"][user_id].add(movie_id)

    state["redis_store"].publish_feedback_event(user_id=user_id, movie_id=movie_id, rating=feedback.rating)

    if feedback.rating >= 4.0:
        state["redis_store"].add_session_like(user_id=user_id, movie_id=movie_id, ttl_seconds=3600)

    return FeedbackResponse(
        status="success",
        message=f"Лайк фильму #{movie_id} записан в Redis. Сессионный контекст User #{user_id} обновлен.",
        event_stream="events:feedback",
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: int,
    k: int = Query(default=10, ge=1, le=50),
    genre: Optional[str] = Query(default=None),
    diversity: float = Query(default=0.0, ge=0.0, le=1.0),
    exploration_rate: float = Query(default=0.1, ge=0.0, le=0.5, description="Доля разведки новинок (Бандит)"),
):
    total_start = time.perf_counter()

    user2idx = state["user2idx"]
    movie2idx = state["movie2idx"]
    if user_id not in user2idx or user_id not in state["user_features"]:
        raise HTTPException(status_code=404, detail=f"User #{user_id} не найден.")

    # 1. Читаем живую сессию из Redis
    t1_start = time.perf_counter()
    recent_likes = state["redis_store"].get_session_likes(user_id=user_id, limit=5)

    uidx = torch.tensor([user2idx[user_id]])
    with torch.no_grad():
        base_user_vec = state["two_tower"].user_tower(uidx)

        if recent_likes:
            recent_indices = [movie2idx[m] for m in recent_likes if m in movie2idx]
            if recent_indices:
                items_tensor = torch.tensor(recent_indices)
                recent_item_vecs = state["two_tower"].item_tower(items_tensor).mean(dim=0, keepdim=True)
                blended = 0.65 * base_user_vec + 0.35 * recent_item_vecs
                user_vector = F.normalize(blended, p=2, dim=1).squeeze(0).tolist()
            else:
                user_vector = base_user_vec.squeeze(0).tolist()
        else:
            user_vector = base_user_vec.squeeze(0).tolist()

    seen_movies = list(state["user_seen"].get(user_id, set()))
    must_not = [models.HasIdCondition(has_id=seen_movies)] if seen_movies else None
    must = [models.FieldCondition(key="genres", match=models.MatchValue(value=genre))] if genre else None

    query_filter = models.Filter(must=must, must_not=must_not)

    qdrant_results = state["qdrant"].query_points(
        collection_name="movies",
        query=user_vector,
        query_filter=query_filter,
        limit=100,
        with_payload=True,
    )
    t1_elapsed = (time.perf_counter() - t1_start) * 1000

    if not qdrant_results.points:
        raise HTTPException(status_code=404, detail="Кандидаты не найдены.")

    # 2. Ранжирование через CatBoost
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

    pool = Pool(X_cand, cat_features=["gender", "genres"])
    probs = state["ranker"].predict_proba(pool)[:, 1]

    for idx, prob in enumerate(probs):
        candidate_rows[idx]["catboost_prob"] = float(prob)

    candidate_rows.sort(key=lambda x: x["catboost_prob"], reverse=True)

    # 3. Разнообразие MMR
    diverse_candidates = apply_mmr_diversity(candidate_rows, top_k=k + 5, lambda_div=diversity)

    # 4. Multi-Armed Bandit Exploration
    final_candidates, explore_count = apply_bandit_exploration(
        diverse_candidates, top_k=k, exploration_rate=exploration_rate
    )
    t2_elapsed = (time.perf_counter() - t2_start) * 1000

    total_elapsed = (time.perf_counter() - total_start) * 1000

    recommendations = []
    for rank_idx, row in enumerate(final_candidates, start=1):
        recommendations.append(
            MovieRecommendation(
                rank=rank_idx,
                movie_id=row["movie_id"],
                title=row["title"],
                year=row["year"],
                genres=row["genres_list"],
                confidence_score=round(row["catboost_prob"], 4),
                retrieval_score=round(float(row["two_tower_score"]), 4),
                is_exploration=row.get("is_bandit_explore", False),
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        total_latency_ms=round(total_elapsed, 2),
        stage1_retrieval_ms=round(t1_elapsed, 2),
        stage2_ranking_ms=round(t2_elapsed, 2),
        candidates_retrieved=len(qdrant_results.points),
        diversity_applied=diversity,
        exploration_rate=exploration_rate,
        exploration_items_count=explore_count,
        session_source="Redis Online Feature Store",
        recommendations=recommendations,
    )