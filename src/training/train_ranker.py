import polars as pl
import numpy as np
import torch
from pathlib import Path
from catboost import CatBoostClassifier, Pool

from src.models.two_tower import TwoTowerModel

def generate_training_data(sample_ratio: float = 0.5):
    print("1. Загрузка взаимодействий и признаков...")
    train_pos = pl.read_parquet("data/processed/train.parquet")
    user_feat = pl.read_parquet("data/processed/user_features.parquet")
    movie_feat = pl.read_parquet("data/processed/movie_features.parquet")
    mappings = torch.load("models/mappings.pt", weights_only=False)
    
    user2idx = mappings["user2idx"]
    movie2idx = mappings["movie2idx"]

    train_pos = train_pos.with_columns(
        pl.col("user_id").cast(pl.Int32),
        pl.col("movie_id").cast(pl.Int32),
    )
    user_feat = user_feat.with_columns(pl.col("user_id").cast(pl.Int32))
    movie_feat = movie_feat.with_columns(pl.col("movie_id").cast(pl.Int32))

    # Несогласованные артефакты нельзя скорить чужим эмбеддингом с индексом 0.
    for column, mapping in (("user_id", user2idx), ("movie_id", movie2idx)):
        missing = set(train_pos[column].to_list()) - mapping.keys()
        if missing:
            raise ValueError(
                f"Training {column} IDs missing from Two-Tower mappings: "
                f"{sorted(missing)[:10]} (total: {len(missing)})"
            )

    # Собираем историю каждого юзера, чтобы НЕ сэмплировать позитивы как негативы
    user_seen_sets = {
        row["user_id"]: set(row["seen"]) 
        for row in train_pos.group_by("user_id").agg(pl.col("movie_id").alias("seen")).iter_rows(named=True)
    }

    train_pos = train_pos.sample(fraction=sample_ratio, seed=42)
    
    # 2. Честная генерация негативных примеров (без пересечений с позитивами юзера)
    print("2. Честная генерация негативных примеров...")
    all_movies = sorted(set(movie_feat["movie_id"].to_list()) & movie2idx.keys())
    rng = np.random.default_rng(42)

    neg_user_ids = []
    neg_movie_ids = []
    
    neg_timestamps = []
    for user_rows in train_pos.partition_by("user_id", maintain_order=True):
        uid = user_rows["user_id"][0]
        seen = user_seen_sets[uid]
        # История взята до сэмплирования позитивов; выборка с возвращением.
        candidates = [mid for mid in all_movies if mid not in seen]
        if not candidates:
            raise ValueError(
                f"No eligible unseen negative movies for user_id={uid}: "
                f"feature catalog / movie2idx intersection has {len(all_movies)} movies"
            )
        neg_user_ids.extend([uid] * len(user_rows))
        neg_movie_ids.extend(rng.choice(candidates, size=len(user_rows)).tolist())
        neg_timestamps.extend(user_rows["timestamp"].to_list())

    pos_df = train_pos.select([
        pl.col("user_id"),
        pl.col("movie_id"),
        pl.col("timestamp"),
        pl.lit(1, dtype=pl.Int32).alias("target"),
    ])

    neg_df = pl.DataFrame(
        {
            "user_id": neg_user_ids,
            "movie_id": neg_movie_ids,
            "timestamp": neg_timestamps,
            "target": [0] * len(neg_user_ids),
        },
        schema={
            "user_id": pl.Int32,
            "movie_id": pl.Int32,
            "timestamp": pl.Int64,
            "target": pl.Int32,
        },
    )

    # Объединяем и сортируем по времени для честного Temporal Split валидации
    full_df = pl.concat([pos_df, neg_df]).sort("timestamp")

    print("3. Объединение признаков (Feature Join)...")
    full_df = full_df.join(user_feat, on="user_id", how="left")
    full_df = full_df.join(movie_feat, on="movie_id", how="left")

    print("4. Расчет Two-Tower скоров...")
    model = TwoTowerModel(num_users=len(user2idx), num_movies=len(movie2idx))
    model.load_state_dict(torch.load("models/two_tower.pt", weights_only=True))
    model.eval()

    u_indices = [user2idx[uid] for uid in full_df["user_id"].to_list()]
    m_indices = [movie2idx[mid] for mid in full_df["movie_id"].to_list()]

    with torch.no_grad():
        u_emb = model.user_tower(torch.tensor(u_indices))
        m_emb = model.item_tower(torch.tensor(m_indices))
        two_tower_scores = (u_emb * m_emb).sum(dim=1).cpu().numpy()

    full_df = full_df.with_columns(
        pl.Series("two_tower_score", two_tower_scores),
        (pl.col("movie_avg_rating") - pl.col("user_avg_rating")).alias("rating_diff"),
    )

    return full_df


def train_catboost_ranker(task_type: str = "CPU"):
    df = generate_training_data(sample_ratio=0.5)

    feature_cols = [
        "gender",
        "age",
        "occupation",
        "user_rating_count",
        "user_avg_rating",
        "release_year",
        "movie_rating_count",
        "movie_avg_rating",
        "movie_std_rating",
        "genres",
        "rating_diff",
        "two_tower_score",
    ]
    cat_features = ["gender", "genres"]

    X = df.select(feature_cols).to_pandas()
    y = df["target"].to_numpy()

    # Заполняем пропуски в строках
    X["gender"] = X["gender"].fillna("Unknown")
    X["genres"] = X["genres"].fillna("Unknown")

    print(f"\nВсего обучающих примеров: {len(X):,}")
    print(f"Признаки: {feature_cols}\n")

    # Делим на train/val
    val_size = int(len(X) * 0.15)
    X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    print(f"5. Обучаем CatBoost (task_type={task_type})...")
    model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.08,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        task_type=task_type,
        verbose=50,
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=40)

    # Сохраняем модель
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model.save_model(models_dir / "ranker.cbm")
    print(f"\nМодель сохранена в {models_dir / 'ranker.cbm'}")

    # Анализ важности признаков (Feature Importance)
    print("\n" + "="*45)
    print("ВАЖНОСТЬ ПРИЗНАКОВ (FEATURE IMPORTANCE):")
    print("="*45)
    importances = model.get_feature_importance(train_pool)
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp / 2)
        print(f"{name:20s} | {imp:6.2f}% {bar}")
    print("="*45 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the CatBoost ranker")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    args = parser.parse_args()
    train_catboost_ranker(task_type=args.task_type)
