import polars as pl
import torch
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm

from metrics import hit_rate_at_k, ndcg_at_k
from model import TwoTowerModel


def evaluate_full_pipeline():
    print("1. Загрузка данных, моделей и признаков...")
    train = pl.read_parquet("data/processed/train.parquet")
    test = pl.read_parquet("data/processed/test.parquet")
    user_feat = pl.read_parquet("data/processed/user_features.parquet")
    movie_feat = pl.read_parquet("data/processed/movie_features.parquet")
    mappings = torch.load("models/mappings.pt", weights_only=False)

    user2idx = mappings["user2idx"]
    movie2idx = mappings["movie2idx"]
    idx2movie = mappings["idx2movie"]

    # Приводим типы
    user_feat = user_feat.with_columns(pl.col("user_id").cast(pl.Int32))
    movie_feat = movie_feat.with_columns(pl.col("movie_id").cast(pl.Int32))

    # Словари для мгновенного лукапа фичей в памяти
    user_dict = {row["user_id"]: row for row in user_feat.iter_rows(named=True)}
    movie_dict = {row["movie_id"]: row for row in movie_feat.iter_rows(named=True)}

    # Загружаем Stage 1 (Two-Tower)
    stage1_model = TwoTowerModel(num_users=len(user2idx), num_movies=len(movie2idx))
    stage1_model.load_state_dict(torch.load("models/two_tower.pt", weights_only=True))
    stage1_model.eval()

    # Загружаем Stage 2 (CatBoost)
    ranker = CatBoostClassifier()
    ranker.load_model("models/ranker.cbm")

    # Предрассчитываем все векторы фильмов
    all_indices = torch.arange(len(movie2idx))
    with torch.no_grad():
        all_item_embeddings = stage1_model.item_tower(all_indices)

    # История просмотров юзеров в Train (маскирование)
    train_history = train.group_by("user_id").agg(pl.col("movie_id").alias("seen"))
    user_seen_dict = {row["user_id"]: [movie2idx[m] for m in row["seen"] if m in movie2idx] for row in train_history.iter_rows(named=True)}

    test_user_items = test.group_by("user_id").agg(pl.col("movie_id").alias("items"))

    hit_rates_10 = []
    ndcgs_10 = []

    feature_cols = [
        "gender", "age", "occupation", "user_rating_count", "user_avg_rating",
        "release_year", "movie_rating_count", "movie_avg_rating", "movie_std_rating",
        "genres", "rating_diff", "two_tower_score"
    ]

    print("2. Прогон полного Two-Stage пайплайна (Two-Tower -> CatBoost) по тесту...")
    for row in tqdm(test_user_items.iter_rows(named=True), total=len(test_user_items)):
        uid = row["user_id"]
        actual_movies = set(row["items"])

        if uid not in user2idx or uid not in user_dict:
            continue

        # --- СТАДИЯ 1: Отбор 100 кандидатов через Two-Tower ---
        uidx = torch.tensor([user2idx[uid]])
        with torch.no_grad():
            u_emb = stage1_model.user_tower(uidx)
            scores = torch.matmul(u_emb, all_item_embeddings.T).squeeze(0)

            seen_indices = user_seen_dict.get(uid, [])
            if seen_indices:
                scores[seen_indices] = -float("inf")

            top100_candidates_idx = torch.topk(scores, k=100).indices.tolist()
            candidate_movie_ids = [idx2movie[idx] for idx in top100_candidates_idx]
            tt_scores = scores[top100_candidates_idx].tolist()

        # --- СТАДИЯ 2: Ранжирование топ-100 кандидатов через CatBoost ---
        u_f = user_dict[uid]
        candidate_rows = []
        for mid, tt_score in zip(candidate_movie_ids, tt_scores):
            m_f = movie_dict.get(mid, None)
            if not m_f:
                continue

            row_data = {
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
                "two_tower_score": tt_score,
                "movie_id": mid,
            }
            candidate_rows.append(row_data)

        cand_df = pl.DataFrame(candidate_rows)
        X_cand = cand_df.select(feature_cols).to_pandas()
        X_cand["gender"] = X_cand["gender"].fillna("Unknown")
        X_cand["genres"] = X_cand["genres"].fillna("Unknown")

        # CatBoost считает вероятности клика P(like)
        cand_pool = Pool(X_cand, cat_features=["gender", "genres"])
        probs = ranker.predict_proba(cand_pool)[:, 1]

        # Сортируем кандидатов по уверенности CatBoost
        cand_df = cand_df.with_columns(pl.Series("catboost_prob", probs))
        reranked_movies = cand_df.sort("catboost_prob", descending=True)["movie_id"].to_list()

        # Замеряем итоговый топ-10
        hit_rates_10.append(hit_rate_at_k(reranked_movies, actual_movies, k=10))
        ndcgs_10.append(ndcg_at_k(reranked_movies, actual_movies, k=10))

    hr10 = sum(hit_rates_10) / len(hit_rates_10)
    ndcg10 = sum(ndcgs_10) / len(ndcgs_10)

    print("\n" + "="*50)
    print("ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ ЭТАПОВ СИСТЕМЫ (K=10):")
    print("="*50)
    print(f"1. Popularity Baseline: HitRate@10 = 0.2463 | NDCG@10 = 0.0396")
    print(f"2. Stage 1 (Two-Tower): HitRate@10 = 0.2026 | NDCG@10 = 0.0309")
    print(f"3. Full Two-Stage:      HitRate@10 = {hr10:.4f} | NDCG@10 = {ndcg10:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    evaluate_full_pipeline()