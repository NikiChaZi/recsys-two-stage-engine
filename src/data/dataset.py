import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class MovieLensDataset(Dataset):
    def __init__(self, users: torch.Tensor, movies: torch.Tensor):
        self.users = users
        self.movies = movies

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx]

def prepare_data_and_mappings(
    train_path: str = "data/processed/train.parquet",
    batch_size: int = 1024,
):
    train_df = pl.read_parquet(train_path)

    # 1. Собираем уникальные ID и создаем плотный маппинг (0 .. N-1)
    unique_users = train_df["user_id"].unique().sort().to_list()
    unique_movies = train_df["movie_id"].unique().sort().to_list()

    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    movie2idx = {mid: idx for idx, mid in enumerate(unique_movies)}

    # 2. Переводим train данные в индексы
    train_df = train_df.with_columns(
        pl.col("user_id").replace_strict(user2idx).alias("user_idx"),
        pl.col("movie_id").replace_strict(movie2idx).alias("movie_idx"),
    )

    users_tensor = torch.tensor(train_df["user_idx"].to_list(), dtype=torch.long)
    movies_tensor = torch.tensor(train_df["movie_idx"].to_list(), dtype=torch.long)

    dataset = MovieLensDataset(users_tensor, movies_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    mappings = {
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        "idx2user": {v: k for k, v in user2idx.items()},
        "idx2movie": {v: k for k, v in movie2idx.items()},
    }

    print(f"Подготовлено данных:")
    print(f"- Уникальных юзеров: {len(user2idx)}")
    print(f"- Уникальных фильмов: {len(movie2idx)}")
    print(f"- Всего обучающих пар: {len(dataset):,}")
    print(f"- Батчей на эпоху (batch_size={batch_size}): {len(dataloader)}")

    return dataloader, mappings

if __name__ == "__main__":
    loader, mappings = prepare_data_and_mappings()
    # Тестовый прогон одного батча
    for users, movies in loader:
        print(f"\nТестовый батч: users shape = {users.shape}, movies shape = {movies.shape}")
        break