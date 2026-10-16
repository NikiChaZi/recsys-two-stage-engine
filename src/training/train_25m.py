import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from src.models.two_tower import TwoTowerModel


def prepare_25m_data(train_path: str = "data/processed_25m/train.parquet"):
    print("1. Чтение 10 млн строк через Polars...")
    t0 = time.perf_counter()
    train_df = pl.read_parquet(train_path)

    print("2. Построение плотных индексов для 162k пользователей и 62k фильмов...")
    unique_users = train_df["user_id"].unique().to_list()
    unique_movies = train_df["movie_id"].unique().to_list()

    user_map_df = pl.DataFrame(
        {"user_id": unique_users, "user_idx": np.arange(len(unique_users), dtype=np.int32)},
        schema={"user_id": pl.Int32, "user_idx": pl.Int32},
    )
    movie_map_df = pl.DataFrame(
        {"movie_id": unique_movies, "movie_idx": np.arange(len(unique_movies), dtype=np.int32)},
        schema={"movie_id": pl.Int32, "movie_idx": pl.Int32},
    )

    train_df = train_df.join(user_map_df, on="user_id", how="left").join(movie_map_df, on="movie_id", how="left")

    user2idx = dict(zip(user_map_df["user_id"].to_list(), user_map_df["user_idx"].to_list()))
    movie2idx = dict(zip(movie_map_df["movie_id"].to_list(), movie_map_df["movie_idx"].to_list()))

    print(f"  - Уникальных пользователей: {len(user2idx):,}")
    print(f"  - Уникальных фильмов:       {len(movie2idx):,}")
    print(f"  - Данные подготовлены за:   {time.perf_counter() - t0:.2f} сек")

    # Превращаем в плоские массивы int64
    u_tensor = torch.from_numpy(train_df["user_idx"].to_numpy().astype(np.int64))
    m_tensor = torch.from_numpy(train_df["movie_idx"].to_numpy().astype(np.int64))

    mappings = {
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        "idx2user": {v: k for k, v in user2idx.items()},
        "idx2movie": {v: k for k, v in movie2idx.items()},
    }
    return u_tensor, m_tensor, mappings


def train_25m(epochs: int = 3, batch_size: int = 4096, lr: float = 0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nЗапуск турбо-обучения Two-Tower на MovieLens 25M...")
    print(f"Используемое устройство: {device}")

    u_tensor, m_tensor, mappings = prepare_25m_data()

    # ВАЖНО: Загружаем ВСЕ 10 млн строк сразу в видеопамять (VRAM занимает всего ~160 МБ!)
    if device.type == "cuda":
        print("Загружаем весь 10M датасет напрямую в VRAM видеокарты...")
        u_tensor = u_tensor.to(device)
        m_tensor = m_tensor.to(device)

    num_samples = u_tensor.size(0)
    num_batches = num_samples // batch_size

    num_users = len(mappings["user2idx"])
    num_movies = len(mappings["movie2idx"])

    model = TwoTowerModel(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=64,
        temperature=0.07,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Всего обучающих батчей на эпоху: {num_batches:,}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # Мгновенная векторная перетасовка индексов прямо в GPU
        perm = torch.randperm(num_samples, device=device)

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{epochs}")
        for b_idx in pbar:
            start_i = b_idx * batch_size
            batch_perm = perm[start_i : start_i + batch_size]

            # Нарезка батча прямо на GPU без участия процессора!
            users = u_tensor[batch_perm]
            movies = m_tensor[batch_perm]

            optimizer.zero_grad(set_to_none=True)
            u_emb, i_emb = model(users, movies)
            loss = model.compute_loss(u_emb, i_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if b_idx % 20 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}/{epochs} | Средний InfoNCE Loss: {avg_loss:.4f}")

    # Сохраняем веса
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / "two_tower_25m.pt")
    torch.save(mappings, models_dir / "mappings_25m.pt")
    print(f"\n✓ Веса модели и маппинги сохранены в {models_dir / 'two_tower_25m.pt'}!")


if __name__ == "__main__":
    train_25m()