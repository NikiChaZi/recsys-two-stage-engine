import os
import re
from pathlib import Path
import polars as pl
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from src.models.two_tower import TwoTowerModel

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"


def parse_25m_movies(movies_csv_path: str = "data/raw/ml-25m/movies.csv") -> dict[int, dict]:
    """Парсит названия, год и жанры из каталога 25M."""
    df = pl.read_csv(movies_csv_path)
    metadata = {}
    for row in df.iter_rows(named=True):
        mid = row["movieId"]
        title_raw = row["title"]
        genres = row["genres"].split("|") if row["genres"] else []

        match = re.search(r"\((\d{4})\)$", title_raw.strip())
        year = int(match.group(1)) if match else None
        clean_title = re.sub(r"\s*\(\d{4}\)$", "", title_raw.strip())

        metadata[mid] = {
            "title": clean_title,
            "year": year,
            "genres": genres,
        }
    return metadata


def index_25m_movies():
    print("1. Подключаемся к Qdrant на localhost:6333...")
    client = QdrantClient(url="http://127.0.0.1:6333", timeout=30)

    print("2. Загружаем обученную модель Two-Tower 25M и маппинги...")
    mappings = torch.load("models/mappings_25m.pt", weights_only=False)
    movie2idx = mappings["movie2idx"]
    idx2movie = mappings["idx2movie"]
    user2idx = mappings["user2idx"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(num_users=len(user2idx), num_movies=len(movie2idx), embedding_dim=64)
    model.load_state_dict(torch.load("models/two_tower_25m.pt", weights_only=True))
    model.to(device)
    model.eval()

    print(f"3. Вычисление эмбеддингов для {len(movie2idx):,} фильмов на {device}...")
    all_indices = torch.arange(len(movie2idx), device=device)
    with torch.no_grad():
        item_vectors = model.item_tower(all_indices).cpu().numpy()

    metadata = parse_25m_movies()

    collection_name = "movies_25m"
    vector_dim = item_vectors.shape[1]

    if client.collection_exists(collection_name):
        print(f"Удаляем старую коллекцию '{collection_name}'...")
        client.delete_collection(collection_name)

    print(f"4. Создаем коллекцию '{collection_name}' (размерность={vector_dim}, метрика=Cosine)...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    print("5. Заливаем векторы и метаданные в Qdrant...")
    batch_size = 500
    points = []

    for idx, mid in idx2movie.items():
        meta = metadata.get(mid, {"title": f"Movie {mid}", "year": None, "genres": []})
        payload = {
            "movie_id": mid,
            "title": meta["title"],
            "year": meta["year"],
            "genres": meta["genres"],
        }
        points.append(
            PointStruct(
                id=mid,
                vector=item_vectors[idx].tolist(),
                payload=payload,
            )
        )

    for i in tqdm(range(0, len(points), batch_size), desc="Индексация 25M в Qdrant"):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    print(f"\n✓ Успех! В коллекцию '{collection_name}' загружено {len(points):,} фильмов!")


if __name__ == "__main__":
    index_25m_movies()