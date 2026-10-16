import os
import time
import polars as pl
import torch
from qdrant_client import QdrantClient
from qdrant_client import models

# Отключаем системный прокси для локального соединения
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"


class QdrantRetrievalService:
    def __init__(self, qdrant_url: str = "http://127.0.0.1:6333"):
        print("Инициализация сервиса поиска...")
        self.client = QdrantClient(url=qdrant_url, timeout=10)
        self.collection_name = "movies"

        # 1. Загрузка маппингов
        mappings = torch.load("models/mappings.pt", weights_only=False)
        self.user2idx = mappings["user2idx"]
        self.movie2idx = mappings["movie2idx"]

        # 2. Загрузка весов обученной модели Two-Tower
        from src.models.two_tower import TwoTowerModel
        self.model = TwoTowerModel(
            num_users=len(self.user2idx),
            num_movies=len(self.movie2idx),
            embedding_dim=64,
        )
        self.model.load_state_dict(torch.load("models/two_tower.pt", weights_only=True))
        self.model.eval()

        # 3. Кэшируем историю просмотров пользователей в Train
        train = pl.read_parquet("data/processed/train.parquet")
        train_hist = train.group_by("user_id").agg(pl.col("movie_id").alias("seen"))
        self.user_history = {row["user_id"]: row["seen"] for row in train_hist.iter_rows(named=True)}
        print("Сервис готов к работе!\n")

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        genre_filter: str | None = None,
    ) -> tuple[list[dict], float]:
        """
        Возвращает топ-K фильмов для пользователя с замером задержки.
        Поддерживает фильтрацию по жанру на лету прямо в Qdrant!
        """
        start_time = time.perf_counter()

        if user_id not in self.user2idx:
            raise ValueError(f"Пользователь {user_id} не найден в обучающей выборке (Cold Start).")

        # 1. Получаем вектор юзера через User Tower
        uidx = torch.tensor([self.user2idx[user_id]])
        with torch.no_grad():
            user_vector = self.model.user_tower(uidx).squeeze(0).tolist()

        # 2. Составляем фильтры для Qdrant
        must_not_conditions = []
        seen_movies = self.user_history.get(user_id, [])
        if seen_movies:
            # Исключаем просмотренные фильмы
            must_not_conditions.append(
                models.HasIdCondition(has_id=seen_movies)
            )

        must_conditions = []
        if genre_filter:
            # Фильтр по жанру на лету: "хочу только комедии"
            must_conditions.append(
                models.FieldCondition(
                    key="genres",
                    match=models.MatchValue(value=genre_filter),
                )
            )

        query_filter = models.Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None,
        )

        # 3. Ультрабыстрый поиск ближайших соседей в Qdrant
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=user_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # 4. Формируем выдачу
        recommendations = []
        for point in search_result.points:
            recommendations.append({
                "movie_id": point.id,
                "title": point.payload.get("title"),
                "year": point.payload.get("year"),
                "genres": point.payload.get("genres"),
                "similarity_score": round(point.score, 4),
            })

        return recommendations, elapsed_ms


if __name__ == "__main__":
    service = QdrantRetrievalService()

    test_user_id = 42

    # Запрос 1: Персональные рекомендации (без фильтра)
    recs, latency = service.recommend(user_id=test_user_id, top_k=10)
    print(f"Персональные рекомендации для User #{test_user_id} (Задержка: {latency:.2f} мс):")
    print("-" * 70)
    for idx, r in enumerate(recs, 1):
        genres_str = ", ".join(r["genres"])
        print(f"{idx:2d}. {r['title']} ({r['year']}) | Скор: {r['similarity_score']} | Жанры: {genres_str}")

    print("\n" + "=" * 70 + "\n")

    # Запрос 2: Персональные рекомендации ТОЛЬКО с жанром Sci-Fi (на лету в базе)
    recs_scifi, latency_scifi = service.recommend(user_id=test_user_id, top_k=5, genre_filter="Sci-Fi")
    print(f"Рекомендации для User #{test_user_id} С ФИЛЬТРОМ [Sci-Fi] (Задержка: {latency_scifi:.2f} мс):")
    print("-" * 70)
    for idx, r in enumerate(recs_scifi, 1):
        genres_str = ", ".join(r["genres"])
        print(f"{idx:2d}. {r['title']} ({r['year']}) | Скор: {r['similarity_score']} | Жанры: {genres_str}")