import polars as pl
import torch
from tqdm import tqdm
from metrics import hit_rate_at_k, ndcg_at_k
from model import TwoTowerModel

def evaluate():
    # 1. Загрузка данных и маппингов
    train = pl.read_parquet("data/processed/train.parquet")
    test = pl.read_parquet("data/processed/test.parquet")
    mappings = torch.load("models/mappings.pt", weights_only=False)
    
    user2idx = mappings["user2idx"]
    movie2idx = mappings["movie2idx"]
    idx2movie = mappings["idx2movie"]
    
    # Собираем историю просмотров в train для каждого юзера
    train_history = (
        train.group_by("user_id")
        .agg(pl.col("movie_id").alias("seen_items"))
    )
    user_seen_dict = {
        row["user_id"]: [movie2idx[m] for m in row["seen_items"] if m in movie2idx]
        for row in train_history.iter_rows(named=True)
    }

    # 2. Загрузка весов
    model = TwoTowerModel(num_users=len(user2idx), num_movies=len(movie2idx))
    model.load_state_dict(torch.load("models/two_tower.pt", weights_only=True))
    model.eval()
    
    # 3. Считаем эмбеддинги ВСЕХ фильмов каталога
    all_movie_indices = torch.arange(len(movie2idx))
    with torch.no_grad():
        all_item_embeddings = model.item_tower(all_movie_indices) # (3456, 64)

    test_user_items = (
        test.group_by("user_id")
        .agg(pl.col("movie_id").alias("items"))
    )

    hit_rates_10 = []
    hit_rates_100 = []
    ndcgs_10 = []
    
    print("Прогоняем честный инференс с маскированием просмотренного...")
    for row in tqdm(test_user_items.iter_rows(named=True), total=len(test_user_items)):
        uid = row["user_id"]
        actual_movies = set(row["items"])
        
        if uid not in user2idx:
            continue
            
        uidx = torch.tensor([user2idx[uid]])
        with torch.no_grad():
            u_emb = model.user_tower(uidx)
            scores = torch.matmul(u_emb, all_item_embeddings.T).squeeze(0) # (3456,)
            
            # ВАЖНО: Зануляем скоры для фильмов, которые юзер уже смотрел в train
            seen_indices = user_seen_dict.get(uid, [])
            if seen_indices:
                scores[seen_indices] = -float("inf")
            
            top100_idx = torch.topk(scores, k=100).indices.tolist()
            recommended_movies = [idx2movie[idx] for idx in top100_idx]

        hit_rates_10.append(hit_rate_at_k(recommended_movies, actual_movies, k=10))
        hit_rates_100.append(hit_rate_at_k(recommended_movies, actual_movies, k=100))
        ndcgs_10.append(ndcg_at_k(recommended_movies, actual_movies, k=10))

    hr10 = sum(hit_rates_10) / len(hit_rates_10)
    hr100 = sum(hit_rates_100) / len(hit_rates_100)
    ndcg10 = sum(ndcgs_10) / len(ndcgs_10)

    print("\n" + "="*45)
    print("Результаты Two-Tower (БЕЗ уже просмотренных фильмов):")
    print(f"HitRate@10:  {hr10:.4f}")
    print(f"HitRate@100: {hr100:.4f}")
    print(f"NDCG@10:     {ndcg10:.4f}")
    print("="*45 + "\n")

if __name__ == "__main__":
    evaluate()