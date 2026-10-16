import ssl
import urllib.request
import zipfile
from pathlib import Path
import polars as pl

DATASET_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def ensure_25m_dataset(raw_dir: Path) -> Path:
    target_dir = raw_dir / "ml-25m"
    ratings_file = target_dir / "ratings.csv"

    if ratings_file.exists():
        return target_dir

    print(f"Скачиваю MovieLens 25M (~250 МБ)... Это займет 1-2 минуты.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-25m.zip"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(DATASET_25M_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())

    print("Распаковываю 25M архив...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    zip_path.unlink()
    print(f"✓ Архив успешно распакован в {target_dir}")
    return target_dir


def process_25m_streaming(raw_dir: str = "data/raw", out_dir: str = "data/processed_25m"):
    raw_path = Path(raw_dir)
    ml_25m_dir = ensure_25m_dataset(raw_path)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n1. Ленивое сканирование (Lazy Scanning) 25 000 000 строк через Polars...")
    
    # scan_csv НЕ читает файл в память, а создает план вычислений
    ratings_lazy = pl.scan_csv(
        ml_25m_dir / "ratings.csv",
        schema_overrides={
            "userId": pl.Int32,
            "movieId": pl.Int32,
            "rating": pl.Float32,
            "timestamp": pl.Int64,
        },
    )

    # 1. Фильтруем позитивные взаимодействия (>= 4.0)
    print("2. Построение плана фильтрации позитивных оценок (>= 4.0)...")
    positive_ratings = ratings_lazy.filter(pl.col("rating") >= 4.0)

    # 2. Чтобы обработать 25M без падения по OOM на персональном компьютере,
    # мы используем Temporal Split по глобальной временной отсечке:
    # 80% времени истории — Train, последние 20% — Test.
    print("3. Расчет квантиля времени для глобального Temporal Split...")
    
    # Считаем 80-й процентиль времени
    time_quantiles = (
        positive_ratings.select(pl.col("timestamp").quantile(0.8).alias("cutoff_time"))
        .collect()
    )
    cutoff_time = time_quantiles["cutoff_time"][0]
    print(f"  - Временная граница отсечки: {cutoff_time}")

    train_lazy = positive_ratings.filter(pl.col("timestamp") <= cutoff_time).select([
        pl.col("userId").alias("user_id"),
        pl.col("movieId").alias("movie_id"),
        pl.col("rating"),
        pl.col("timestamp"),
    ])

    test_lazy = positive_ratings.filter(pl.col("timestamp") > cutoff_time).select([
        pl.col("userId").alias("user_id"),
        pl.col("movieId").alias("movie_id"),
        pl.col("rating"),
        pl.col("timestamp"),
    ])

    # 3. Запуск потокового сохранения в Parquet (Streaming Execution)
    print("4. Стриминг Train набора в Parquet...")
    train_lazy.sink_parquet(output_path / "train.parquet", compression="zstd")

    print("5. Стриминг Test набора в Parquet...")
    test_lazy.sink_parquet(output_path / "test.parquet", compression="zstd")

    # 4. Проверяем финальный объем
    train_count = pl.scan_parquet(output_path / "train.parquet").select(pl.len()).collect()["len"][0]
    test_count = pl.scan_parquet(output_path / "test.parquet").select(pl.len()).collect()["len"][0]

    print("\n" + "=" * 50)
    print("ИТОГИ ОБРАБОТКИ MOVIELENS 25M:")
    print("=" * 50)
    print(f"Train взаимодействий: {train_count:,}")
    print(f"Test взаимодействий:  {test_count:,}")
    print(f"Всего сохранено:      {train_count + test_count:,}")
    print(f"Папка с данными:      {output_path.resolve()}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    process_25m_streaming()