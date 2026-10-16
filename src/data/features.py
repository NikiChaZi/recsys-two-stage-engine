import re
from pathlib import Path
import polars as pl


def extract_user_features(raw_dir: str = "data/raw/ml-1m") -> pl.DataFrame:
    """Вытаскивает демографию и активность пользователей."""
    raw_path = Path(raw_dir)
    with open(raw_path / "users.dat", "rb") as f:
        users_bytes = f.read().replace(b"::", b"\t")

    users_df = pl.read_csv(
        users_bytes,
        separator="\t",
        has_header=False,
        new_columns=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="iso-8859-1",
        schema_overrides={"zip_code": pl.String},  # <-- Защита от дефисов в индексах
    ).drop("zip_code")

    # Считаем активность и средний рейтинг юзера по Train
    train_df = pl.read_parquet("data/processed/train.parquet")
    user_stats = train_df.group_by("user_id").agg(
        pl.len().alias("user_rating_count"),
        pl.col("rating").mean().alias("user_avg_rating"),
    )

    return users_df.join(user_stats, on="user_id", how="left").with_columns(
        pl.col("user_rating_count").fill_null(0),
        pl.col("user_avg_rating").fill_null(3.0),
    )


def extract_movie_features(raw_dir: str = "data/raw/ml-1m") -> pl.DataFrame:
    """Вытаскивает год, жанры и статистику популярности фильмов."""
    raw_path = Path(raw_dir)
    with open(raw_path / "movies.dat", "rb") as f:
        movies_bytes = f.read().replace(b"::", b"\t")

    movies_df = pl.read_csv(
        movies_bytes,
        separator="\t",
        has_header=False,
        new_columns=["movie_id", "title", "genres"],
        encoding="iso-8859-1",
    )

    # Парсим год из названия
    years = []
    for title in movies_df["title"]:
        match = re.search(r"\((\d{4})\)$", title.strip())
        years.append(int(match.group(1)) if match else 1990)

    movies_df = movies_df.with_columns(pl.Series("release_year", years))

    # Считаем статистику фильмов по Train
    train_df = pl.read_parquet("data/processed/train.parquet")
    movie_stats = train_df.group_by("movie_id").agg(
        pl.len().alias("movie_rating_count"),
        pl.col("rating").mean().alias("movie_avg_rating"),
        pl.col("rating").std().alias("movie_std_rating"),
    )

    return movies_df.join(movie_stats, on="movie_id", how="left").with_columns(
        pl.col("movie_rating_count").fill_null(0),
        pl.col("movie_avg_rating").fill_null(3.0),
        pl.col("movie_std_rating").fill_null(0.0),
    )


if __name__ == "__main__":
    print("Генерируем признаки пользователей...")
    u_feat = extract_user_features()
    print(f"User Features: {u_feat.shape}")
    print(u_feat.head(3))

    print("\nГенерируем признаки фильмов...")
    m_feat = extract_movie_features()
    print(f"Movie Features: {m_feat.shape}")
    print(m_feat.head(3))

    # Сохраняем в processed
    out_dir = Path("data/processed")
    u_feat.write_parquet(out_dir / "user_features.parquet")
    m_feat.write_parquet(out_dir / "movie_features.parquet")
    print(f"\nПризнаки сохранены в {out_dir.resolve()}!")