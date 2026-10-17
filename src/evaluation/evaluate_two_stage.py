import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import polars as pl
import torch
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm

from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k
from src.models.two_tower import TwoTowerModel


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

    user_feat = user_feat.with_columns(pl.col("user_id").cast(pl.Int32))
    movie_feat = movie_feat.with_columns(pl.col("movie_id").cast(pl.Int32))

    user_dict = {row["user_id"]: row for row in user_feat.iter_rows(named=True)}
    movie_dict = {row["movie_id"]: row for row in movie_feat.iter_rows(named=True)}

    # Загружаем Stage 1 (Two-Tower)
    stage1_model = TwoTowerModel(num_users=len(user2idx), num_movies=len(movie2idx))
    stage1_model.load_state_dict(torch.load("models/two_tower.pt", weights_only=True))
    stage1_model.eval()

    # Загружаем Stage 2 (CatBoost)
    ranker = CatBoostClassifier()
    ranker.load_model("models/ranker.cbm")

    all_indices = torch.arange(len(movie2idx))
    with torch.no_grad():
        all_item_embeddings = stage1_model.item_tower(all_indices)

    # История просмотров: разделяем сырые movie_id и индексы!
    train_history = train.group_by("user_id").agg(pl.col("movie_id").alias("seen"))
    user_seen_raw_ids = {row["user_id"]: set(row["seen"]) for row in train_history.iter_rows(named=True)}
    user_seen_idx = {row["user_id"]: [movie2idx[m] for m in row["seen"] if m in movie2idx] for row in train_history.iter_rows(named=True)}

    # Глобальный бейзлайн популярности (топ-200, чтобы хватило после фильтрации просмотренного)
    popular_movies = (
        train.group_by("movie_id").len().sort("len", descending=True).limit(200)["movie_id"].to_list()
    )

    test_user_items = test.group_by("user_id").agg(pl.col("movie_id").alias("items"))

    baseline_hr, baseline_ndcg = [], []
    stage1_hr, stage1_ndcg, stage1_hr100 = [], [], []
    stage2_hr, stage2_ndcg = [], []

    feature_cols = [
        "gender", "age", "occupation", "user_rating_count", "user_avg_rating",
        "release_year", "movie_rating_count", "movie_avg_rating", "movie_std_rating",
        "genres", "rating_diff", "two_tower_score"
    ]

    print("2. Динамический расчет ВСЕХ 3 моделей на одном отложенном тесте...")
    for row in tqdm(test_user_items.iter_rows(named=True), total=len(test_user_items)):
        uid = row["user_id"]
        actual_movies = set(row["items"])

        if uid not in user2idx or uid not in user_dict:
            continue

        raw_seen = user_seen_raw_ids.get(uid, set())
        idx_seen = user_seen_idx.get(uid, [])

        # 1. ЧЕСТНЫЙ бейзлайн популярности (сравниваем movie_id с movie_id!)
        pop_unseen = [m for m in popular_movies if m not in raw_seen][:10]
        baseline_hr.append(hit_rate_at_k(pop_unseen, actual_movies, k=10))
        baseline_ndcg.append(ndcg_at_k(pop_unseen, actual_movies, k=10))

        # 2. Stage 1: Two-Tower (маскируем по индексам матрицы)
        uidx = torch.tensor([user2idx[uid]])
        with torch.no_grad():
            u_emb = stage1_model.user_tower(uidx)
            scores = torch.matmul(u_emb, all_item_embeddings.T).squeeze(0)

            if idx_seen:
                scores[idx_seen] = -float("inf")

            top100_idx = torch.topk(scores, k=100).indices.tolist()
            stage1_candidates = [idx2movie[idx] for idx in top100_idx]
            tt_scores = scores[top100_idx].tolist()

        stage1_hr.append(hit_rate_at_k(stage1_candidates[:10], actual_movies, k=10))
        stage1_hr100.append(hit_rate_at_k(stage1_candidates, actual_movies, k=100))
        stage1_ndcg.append(ndcg_at_k(stage1_candidates[:10], actual_movies, k=10))

        # 3. Stage 2: CatBoost Re-Ranker (ОДИН РАЗ, БЕЗ ДУБЛЯ)
        u_f = user_dict[uid]
        candidate_rows = []
        for mid, tt_score in zip(stage1_candidates, tt_scores):
            m_f = movie_dict.get(mid)
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
                "two_tower_score": tt_score,
                "movie_id": mid,
            })

        cand_df = pl.DataFrame(candidate_rows)
        X_cand = cand_df.select(feature_cols).to_pandas()
        X_cand["gender"] = X_cand["gender"].fillna("Unknown")
        X_cand["genres"] = X_cand["genres"].fillna("Unknown")

        pool = Pool(X_cand, cat_features=["gender", "genres"])
        probs = ranker.predict_proba(pool)[:, 1]

        cand_df = cand_df.with_columns(pl.Series("prob", probs))
        stage2_top10 = cand_df.sort("prob", descending=True)["movie_id"].head(10).to_list()

        stage2_hr.append(hit_rate_at_k(stage2_top10, actual_movies, k=10))
        stage2_ndcg.append(ndcg_at_k(stage2_top10, actual_movies, k=10))

    # Считаем средние
    results = {
        "popularity_baseline": {
            "hit_rate_at_10": round(sum(baseline_hr) / len(baseline_hr), 4),
            "ndcg_at_10": round(sum(baseline_ndcg) / len(baseline_ndcg), 4),
        },
        "stage1_two_tower": {
            "hit_rate_at_10": round(sum(stage1_hr) / len(stage1_hr), 4),
            "hit_rate_at_100": round(sum(stage1_hr100) / len(stage1_hr100), 4),
            "ndcg_at_10": round(sum(stage1_ndcg) / len(stage1_ndcg), 4),
        },
        "stage2_full_pipeline": {
            "hit_rate_at_10": round(sum(stage2_hr) / len(stage2_hr), 4),
            "ndcg_at_10": round(sum(stage2_ndcg) / len(stage2_ndcg), 4),
        }
    }

    res_path = Path("data/processed/evaluation_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*55)
    print("ИТОГОВОЕ ЧЕСТНОЕ СРАВНЕНИЕ (БЕЗ ДУБЛЕЙ И БЕЗ БАГОВ):")
    print("="*55)
    print(f"1. Baseline:   HitRate@10 = {results['popularity_baseline']['hit_rate_at_10']} | NDCG@10 = {results['popularity_baseline']['ndcg_at_10']}")
    print(f"2. Stage 1:    HitRate@10 = {results['stage1_two_tower']['hit_rate_at_10']} | HitRate@100 = {results['stage1_two_tower']['hit_rate_at_100']} | NDCG@10 = {results['stage1_two_tower']['ndcg_at_10']}")
    print(f"3. Two-Stage:  HitRate@10 = {results['stage2_full_pipeline']['hit_rate_at_10']} | NDCG@10 = {results['stage2_full_pipeline']['ndcg_at_10']}")
    print(f"✓ Артефакт сохранен в: {res_path}")
    print("="*55 + "\n")


if __name__ == "__main__":
    evaluate_full_pipeline()