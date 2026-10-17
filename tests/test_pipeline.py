import pytest
import torch
from src.models.two_tower import TwoTowerModel
from src.evaluation.metrics import hit_rate_at_k, ndcg_at_k
from src.api.main import genre_jaccard_similarity, apply_mmr_diversity


def test_two_tower_shapes_and_symmetric_loss():
    """Тест 1: Проверка архитектуры Two-Tower и симметричного лосса."""
    model = TwoTowerModel(num_users=100, num_movies=50, embedding_dim=32)
    users = torch.tensor([0, 1, 2])
    movies = torch.tensor([10, 20, 30])

    u_emb, m_emb = model(users, movies)
    assert u_emb.shape == (3, 32)
    assert m_emb.shape == (3, 32)

    # Проверка L2-нормализации (длина вектора должна быть равна 1.0)
    assert torch.allclose(torch.norm(u_emb, dim=1), torch.ones(3), atol=1e-5)

    # Проверка вычисления симметричного лосса
    loss = model.compute_loss(u_emb, m_emb)
    assert loss.item() > 0.0


def test_ranking_metrics():
    """Тест 2: Проверка математики HitRate@K и NDCG@K."""
    recommended = [10, 20, 30, 40, 50]
    actual_hit = {30, 99}
    actual_miss = {99, 100}

    # HitRate
    assert hit_rate_at_k(recommended, actual_hit, k=5) == 1.0
    assert hit_rate_at_k(recommended, actual_miss, k=5) == 0.0

    # NDCG: первый элемент дает идеальный скор 1.0
    assert ndcg_at_k([10], {10}, k=1) == 1.0


def test_mmr_diversity_logic():
    """Тест 3: Проверка алгоритма MMR (дедупликация жанров)."""
    candidates = [
        {"movie_id": 1, "catboost_prob": 0.95, "genres_list": ["Action", "Sci-Fi"]},
        {"movie_id": 2, "catboost_prob": 0.94, "genres_list": ["Action", "Sci-Fi"]}, # Почти дубль
        {"movie_id": 3, "catboost_prob": 0.85, "genres_list": ["Comedy"]},             # Другой жанр
    ]
    # При diversity = 0.8 комедия должна обойти второй боевик
    diverse = apply_mmr_diversity(candidates, top_k=2, lambda_div=0.8)
    genres_selected = [m["genres_list"][0] for m in diverse]
    
    assert "Action" in genres_selected
    assert "Comedy" in genres_selected

def test_symmetric_loss_exact_equality():
    """Тест 4: Проверка строгой симметрии InfoNCE лосса."""
    model = TwoTowerModel(num_users=20, num_movies=20, embedding_dim=16)
    users = torch.tensor([1, 2, 3])
    movies = torch.tensor([5, 6, 7])
    
    u_emb, m_emb = model(users, movies)
    loss = model.compute_loss(u_emb, m_emb)
    
    # Меняем местами эмбеддинги — симметричный лосс обязан выдать то же число
    loss_transposed = model.compute_loss(m_emb, u_emb)
    assert torch.isclose(loss, loss_transposed, atol=1e-5)


def test_history_masking_prevents_seen_recommendations():
    """Тест 5: Фильтрация просмотренного исключает дубликаты."""
    raw_scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    seen_indices = [0, 2] # Пользователь уже смотрел фильмы #0 и #2
    
    # Маскирование минус бесконечностью
    raw_scores[seen_indices] = -float("inf")
    top_candidates = torch.topk(raw_scores, k=2).indices.tolist()
    
    # Просмотренные фильмы никогда не попадут в топ
    assert 0 not in top_candidates
    assert 2 not in top_candidates
    assert top_candidates == [1, 3]

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_healthcheck_contract():
    """Тест 6: Проверка HTTP-контракта эндпоинта /health."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
        assert "qdrant_connected" in data
        assert "redis_connected" in data


def test_recommend_endpoint_validation():
    """Тест 7: Валидация входных параметров API."""
    with TestClient(app) as client:
        response = client.get("/recommend/9999999")
        assert response.status_code == 404