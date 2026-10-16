import random
from locust import HttpUser, task, between

class RecSysUser(HttpUser):
    # Пользователь делает паузу от 0.1 до 0.5 сек между кликами
    wait_time = between(0.1, 0.5)

    @task(5)
    def get_recommendations_pure(self):
        """80% пользователей просто листают персональную ленту."""
        uid = random.randint(1, 6040)
        self.client.get(f"/recommend/{uid}?k=10", name="/recommend/[user_id]")

    @task(2)
    def get_recommendations_with_diversity(self):
        """20% пользователей запрашивают ленту с повышенным разнообразием."""
        uid = random.randint(1, 6040)
        diversity = round(random.uniform(0.3, 0.7), 2)
        self.client.get(
            f"/recommend/{uid}?k=10&diversity={diversity}",
            name="/recommend/[user_id]?diversity",
        )

    @task(1)
    def send_feedback(self):
        """Редкие пользователи ставят лайки прямо во время сессии."""
        uid = random.randint(1, 6040)
        mid = random.randint(1, 3500)
        rating = random.choice([4.0, 4.5, 5.0])
        payload = {"user_id": uid, "movie_id": mid, "rating": rating}
        self.client.post("/feedback", json=payload, name="/feedback")