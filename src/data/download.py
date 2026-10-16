import ssl
import urllib.request
import zipfile
from pathlib import Path
import polars as pl

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

def ensure_dataset(raw_dir: Path) -> Path:
    """Проверяет наличие датасета, если нет — скачивает и распаковывает."""
    target_dir = raw_dir / "ml-1m"
    ratings_file = target_dir / "ratings.dat"
    
    if ratings_file.exists():
        return target_dir

    print(f"Датасет не найден. Скачиваю MovieLens 1M (~6 МБ)...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-1m.zip"

    # Фикс SSL для Windows
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())

    print("Распаковываю архив...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    zip_path.unlink()
    print(f"Датасет успешно распакован в {target_dir}")
    return target_dir

def load_and_preprocess_data(raw_dir: str = "data/raw") -> pl.DataFrame:
    raw_path = Path(raw_dir)
    ml_1m_dir = ensure_dataset(raw_path)
    ratings_file = ml_1m_dir / "ratings.dat"

    # Читаем и заменяем разделитель '::' на табуляцию '\t'
    with open(ratings_file, "rb") as f:
        data_bytes = f.read().replace(b"::", b"\t")

    ratings = pl.read_csv(
        data_bytes,
        separator="\t",
        has_header=False,
        new_columns=["user_id", "movie_id", "rating", "timestamp"],
        schema_overrides={
            "user_id": pl.Int32,
            "movie_id": pl.Int32,
            "rating": pl.Float32,
            "timestamp": pl.Int64,
        },
    )

    # Оставляем позитивные взаимодействия (рейтинг >= 4.0)
    positive_ratings = ratings.filter(pl.col("rating") >= 4.0)
    return positive_ratings.sort(["user_id", "timestamp"])

def temporal_split(df: pl.DataFrame, test_ratio: float = 0.2):
    """
    Делим взаимодействия каждого пользователя во времени:
    ранние 80% идут в train, последние 20% — в test.
    """
    df = df.with_columns(
        pl.col("timestamp").rank(method="ordinal").over("user_id").alias("rank"),
        pl.len().over("user_id").alias("user_total_interactions")  # <-- здесь pl.len()
    )

    df = df.with_columns(
        (pl.col("user_total_interactions") * (1.0 - test_ratio)).floor().alias("train_cutoff")
    )

    train = df.filter(pl.col("rank") <= pl.col("train_cutoff")).drop(["rank", "user_total_interactions", "train_cutoff"])
    test = df.filter(pl.col("rank") > pl.col("train_cutoff")).drop(["rank", "user_total_interactions", "train_cutoff"])

    return train, test

if __name__ == "__main__":
    df = load_and_preprocess_data()
    train, test = temporal_split(df)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train.write_parquet(output_dir / "train.parquet")
    test.write_parquet(output_dir / "test.parquet")

    print(f"\nУспех! Train строк: {len(train):,}, Test строк: {len(test):,}")