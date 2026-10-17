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