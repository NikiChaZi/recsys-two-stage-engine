"""Feedback consistency checks without model lifecycle or a Redis server."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError, ResponseError, TimeoutError

from src.api import main as api
from src.services.redis_store import RedisSessionStore


@pytest.fixture
def feedback_client(monkeypatch):
    store = MagicMock(spec=RedisSessionStore)
    isolated_state = {"user_seen": {10: {100}}, "redis_store": store}
    monkeypatch.setattr(api, "state", isolated_state)
    # No context manager: Starlette does not run the model/service lifespan.
    client = TestClient(api.app)
    yield client, store, isolated_state
    client.close()


@pytest.mark.parametrize("user_id", [10, 20])
@pytest.mark.parametrize("rating", [5.0, 2.0])
def test_feedback_updates_history_only_after_confirmed_write(feedback_client, user_id, rating):
    client, store, state = feedback_client
    before = deepcopy(state["user_seen"])

    def confirm_write(**kwargs):
        assert state["user_seen"] == before

    store.record_feedback.side_effect = confirm_write
    response = client.post("/feedback", json={"user_id": user_id, "movie_id": 200, "rating": rating})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["event_stream"] == "events:feedback"
    assert ("Сессионный контекст" in body["message"]) == (rating >= 4.0)
    assert "Лайк" not in body["message"]
    before.setdefault(user_id, set()).add(200)
    assert state["user_seen"] == before
    store.record_feedback.assert_called_once_with(
        user_id=user_id, movie_id=200, rating=rating, ttl_seconds=3600
    )
    store.publish_feedback_event.assert_not_called()
    store.add_session_like.assert_not_called()


@pytest.mark.parametrize("user_id", [10, 20])
@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError, ResponseError])
def test_redis_failure_returns_503_without_changing_history(feedback_client, user_id, error_type):
    client, store, state = feedback_client
    before = deepcopy(state["user_seen"])
    store.record_feedback.side_effect = error_type("Redis write failed")
    response = client.post("/feedback", json={"user_id": user_id, "movie_id": 200, "rating": 5.0})
    assert response.status_code == 503
    assert "Не удалось подтвердить" in response.json()["detail"]
    assert "status" not in response.json()
    assert state["user_seen"] == before


@pytest.fixture
def transactional_store():
    # Skip connection construction entirely; record only the Redis command contract.
    store = RedisSessionStore.__new__(RedisSessionStore)
    store.r = MagicMock()
    pipe = store.r.pipeline.return_value.__enter__.return_value
    return store, pipe


@pytest.mark.parametrize("rating", [5.0, 4.0, 3.9])
def test_feedback_event_and_session_commands_share_one_transaction(transactional_store, monkeypatch, rating):
    store, pipe = transactional_store
    monkeypatch.setattr("src.services.redis_store.time.time", lambda: 1234)
    store.record_feedback(user_id=10, movie_id=200, rating=rating, ttl_seconds=120)
    store.r.pipeline.assert_called_once_with(transaction=True)
    calls = [(call[0], call[1], call[2]) for call in pipe.method_calls]
    expected = [("xadd", ("events:feedback", {
        "user_id": "10", "movie_id": "200", "rating": str(rating), "timestamp": "1234"
    }), {"maxlen": 50000})]
    if rating >= 4.0:
        expected.extend([
            ("lpush", ("session:10", 200), {}),
            ("ltrim", ("session:10", 0, 9), {}),
            ("expire", ("session:10", 120), {}),
        ])
    expected.append(("execute", (), {}))
    assert calls == expected
    store.r.xadd.assert_not_called()


def test_transaction_error_propagates_without_retry(transactional_store):
    store, pipe = transactional_store
    pipe.execute.side_effect = ConnectionError("EXEC reply lost")
    with pytest.raises(ConnectionError, match="EXEC reply lost"):
        store.record_feedback(user_id=10, movie_id=200, rating=5.0)
    pipe.execute.assert_called_once_with()
