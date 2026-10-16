import os
# Отключаем системный прокси для локального хоста
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import re
import polars as pl
import torch
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

from model import TwoTowerModel

def parse_movies_metadata(raw_path: str = "data/raw/ml-1m/movies.dat") -> dict[int, dict]:
    """Парсит названия, год и жанры фильмов."""
    with open(raw_path, "rb") as f:
        data_bytes = f.read().replace(b"::", b"\t")

    movies_df = pl.read_csv(
        data_bytes,
        separator="\t",
        has_header=False,
        new_columns=["movie_id", "title", "genres"],
        encoding="iso-8859-1",
    )

    metadata = {}
    for row in movies_df.iter_rows(named=True):
        mid = row["movie_id"]
        title_raw = row["title"]
        genres = row["genres"].split("|")
        
        # Вытаскиваем год из названия: "Toy Story (1995)" -> 1995
        year_match = re.search(r"\((\d{4})\)$", title_raw.strip())
        year = int(year_match.group(1)) if year_match else None
        clean_title = re.sub(r"\s*\(\d{4}\)$", "", title_raw.strip())

        metadata[mid] = {
            "title": clean_title,
            "year": year,
            "genres": genres,
        }
    return metadata

def index_movies():
    print("1. Подключаемся к Qdrant на localhost:6333...")
    client = QdrantClient(url="http://127.0.0.1:6333", timeout=30)

    print("2. Загружаем маппинги и обученную модель Two-Tower...")
    mappings = torch.load("models/mappings.pt", weights_only=False)
    movie2idx = mappings["movie2idx"]
    idx2movie = mappings["idx2movie"]
    user2idx = mappings["user2idx"]

    # Инициализация и загрузка весов
    model = TwoTowerModel(num_users=len(user2idx), num_movies=len(movie2idx), embedding_dim=64)
    model.load_state_dict(torch.load("models/two_tower.pt", weights_only=True))
    model.eval()

    print("3. Считаем эмбеддинги фильмов через Item Tower...")
    all_indices = torch.arange(len(movie2idx))
    with torch.no_grad():
        item_vectors = model.item_tower(all_indices).cpu().numpy()

    # Загружаем метаданные
    metadata = parse_movies_metadata()

    collection_name = "movies"
    vector_dim = item_vectors.shape[1]

    # Пересоздаем коллекцию, если уже была
    if client.collection_exists(collection_name):
        print(f"Удаляем старую коллекцию '{collection_name}'...")
        client.delete_collection(collection_name)

    print(f"4. Создаем коллекцию '{collection_name}' (размерность={vector_dim}, метрика=Cosine)...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    print("5. Заливаем векторы и метаданные в Qdrant...")
    batch_size = 256
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
                id=mid,  # в качестве ID точки используем оригинальный movie_id
                vector=item_vectors[idx].tolist(),
                payload=payload,
            )
        )

    # Загружаем пачками
    for i in tqdm(range(0, len(points), batch_size), desc="Индексация в Qdrant"):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    print(f"\nУспех! В коллекцию '{collection_name}' загружено {len(points)} фильмов.")

if __name__ == "__main__":
    index_movies()