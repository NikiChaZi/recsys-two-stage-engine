import polars as pl
from metrics import hit_rate_at_k, ndcg_at_k
from tqdm import tqdm

def run_popularity_baseline():
    train = pl.read_parquet("data/processed/train.parquet")
    test = pl.read_parquet("data/processed/test.parquet")
    
    # Считаем самые популярные фильмы в train через .len()
    popular_movies = (
        train.group_by("movie_id")
        .len()
        .sort("len", descending=True)
        .limit(100)["movie_id"]
        .to_list()
    )
    
    # Группируем тестовые фильмы по пользователям
    test_user_items = (
        test.group_by("user_id")
        .agg(pl.col("movie_id").alias("items"))
    )
    
    hit_rates = []
    ndcgs = []
    k = 10
    
    for row in tqdm(test_user_items.iter_rows(named=True), total=len(test_user_items)):
        actual_set = set(row["items"])
        hit_rates.append(hit_rate_at_k(popular_movies, actual_set, k=k))
        ndcgs.append(ndcg_at_k(popular_movies, actual_set, k=k))
        
    mean_hr = sum(hit_rates) / len(hit_rates)
    mean_ndcg = sum(ndcgs) / len(ndcgs)
    
    print("\n" + "="*35)
    print(f"Popularity Baseline Results (K={k}):")
    print(f"HitRate@{k}: {mean_hr:.4f}")
    print(f"NDCG@{k}:    {mean_ndcg:.4f}")
    print("="*35 + "\n")

if __name__ == "__main__":
    run_popularity_baseline()