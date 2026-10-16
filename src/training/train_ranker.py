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

    # Приводим ID к единому типу Int32 во всех таблицах
    train_pos = train_pos.with_columns(
        pl.col("user_id").cast(pl.Int32),
        pl.col("movie_id").cast(pl.Int32),
    )
    user_feat = user_feat.with_columns(pl.col("user_id").cast(pl.Int32))
    movie_feat = movie_feat.with_columns(pl.col("movie_id").cast(pl.Int32))

    # Для быстрой скорости обучения возьмем репрезентативную подвыборку
    train_pos = train_pos.sample(fraction=sample_ratio, seed=42)
    users_sample = train_pos["user_id"].to_list()
    
    # 2. Генерируем негативные примеры (y = 0)
    print("2. Генерация негативных примеров...")
    all_movies = movie_feat["movie_id"].to_list()
    rng = np.random.default_rng(42)
    random_neg_movies = rng.choice(all_movies, size=len(train_pos))

    pos_df = train_pos.select([
        pl.col("user_id"),
        pl.col("movie_id"),
        pl.lit(1, dtype=pl.Int32).alias("target"),
    ])

    neg_df = pl.DataFrame(
        {
            "user_id": users_sample,
            "movie_id": [int(m) for m in random_neg_movies],
            "target": [0] * len(train_pos),
        },
        schema={
            "user_id": pl.Int32,
            "movie_id": pl.Int32,
            "target": pl.Int32,
        },
    )

    full_df = pl.concat([pos_df, neg_df]).sample(fraction=1.0, shuffle=True, seed=42)

    # 3. Джойним фичи юзера и фильма
    print("3. Объединение признаков (Feature Join)...")
    full_df = full_df.join(user_feat, on="user_id", how="left")
    full_df = full_df.join(movie_feat, on="movie_id", how="left")

    # 4. Считаем скор Two-Tower модели для каждой пары
    print("4. Расчет Two-Tower скоров...")
    model = TwoTowerModel(num_users=len(user2idx), num_movies=len(movie2idx))
    model.load_state_dict(torch.load("models/two_tower.pt", weights_only=True))
    model.eval()

    u_indices = [user2idx.get(uid, 0) for uid in full_df["user_id"].to_list()]
    m_indices = [movie2idx.get(mid, 0) for mid in full_df["movie_id"].to_list()]

    with torch.no_grad():
        u_emb = model.user_tower(torch.tensor(u_indices))
        m_emb = model.item_tower(torch.tensor(m_indices))
        two_tower_scores = (u_emb * m_emb).sum(dim=1).cpu().numpy()

    full_df = full_df.with_columns(
        pl.Series("two_tower_score", two_tower_scores),
        (pl.col("movie_avg_rating") - pl.col("user_avg_rating")).alias("rating_diff"),
    )

    return full_df


def train_catboost_ranker():
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

    print("5. Обучаем CatBoost...")
    model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.08,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        task_type="CPU",
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
    train_catboost_ranker()