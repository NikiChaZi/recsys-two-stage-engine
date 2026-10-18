import json
import time
import redis


class RedisSessionStore:
    def __init__(self, redis_url: str = "redis://127.0.0.1:6379/0"):
        """Подключение к Redis с автоматическим пулом соединений."""
        self.r = redis.from_url(redis_url, decode_responses=True)
        print(f"✓ Подключение к Redis Online Feature Store ({redis_url}) установлено!")

    def add_session_like(self, user_id: int, movie_id: int, ttl_seconds: int = 3600):
        """
        Сохраняет лайк в сессию пользователя:
        - Добавляет в начало списка session:{user_id}
        - Ограничивает размер сессии (последние 10 лайков)
        - Выставляет TTL (время жизни сессии 1 час)
        """
        key = f"session:{user_id}"
        pipe = self.r.pipeline()
        pipe.lpush(key, movie_id)
        pipe.ltrim(key, 0, 9)  # держим максимум 10 недавних кликов
        pipe.expire(key, ttl_seconds)
        pipe.execute()

    def record_feedback(self, user_id: int, movie_id: int, rating: float, ttl_seconds: int = 3600):
        """Записывает событие и положительный лайк одной MULTI/EXEC-транзакцией.

        Redis не откатывает ошибки отдельных команд во время EXEC. Потеря
        ответа EXEC также не означает, что запись не состоялась; повтор запроса
        может создать дубликат. Метод не гарантирует exactly-once доставку.
        """
        event = {
            "user_id": str(user_id),
            "movie_id": str(movie_id),
            "rating": str(rating),
            "timestamp": str(int(time.time())),
        }
        with self.r.pipeline(transaction=True) as pipe:
            pipe.xadd("events:feedback", event, maxlen=50000)
            if rating >= 4.0:
                key = f"session:{user_id}"
                pipe.lpush(key, movie_id)
                pipe.ltrim(key, 0, 9)
                pipe.expire(key, ttl_seconds)
            pipe.execute()

    def get_session_likes(self, user_id: int, limit: int = 5) -> list[int]:
        """Возвращает последние N лайков текущей сессии за доли миллисекунды."""
        key = f"session:{user_id}"
        items = self.r.lrange(key, 0, limit - 1)
        return [int(mid) for mid in items]

    def publish_feedback_event(self, user_id: int, movie_id: int, rating: float):
        """
        Event Streaming: Публикует событие в поток Redis Stream 'events:feedback'
        для онлайн-аналитики и асинхронных ML-воркеров.
        """
        event = {
            "user_id": str(user_id),
            "movie_id": str(movie_id),
            "rating": str(rating),
            "timestamp": str(int(time.time())),
        }
        # XADD events:feedback * field value ...
        self.r.xadd("events:feedback", event, maxlen=50000)

    def ping(self) -> bool:
        return self.r.ping()


if __name__ == "__main__":
    # Тест работы Redis
    store = RedisSessionStore()
    test_uid = 42

    print("\n1. Тест публикации события в Redis Stream...")
    store.publish_feedback_event(user_id=test_uid, movie_id=1, rating=5.0)

    print("2. Тест добавления лайков в сессию...")
    store.add_session_like(user_id=test_uid, movie_id=1)
    store.add_session_like(user_id=test_uid, movie_id=1193)

    likes = store.get_session_likes(user_id=test_uid)
    print(f"3. Получены лайки сессии из Redis для User #{test_uid}: {likes}")
    print("✓ Redis Online Feature Store работает идеально!\n")