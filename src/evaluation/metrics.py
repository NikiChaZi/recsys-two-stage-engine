import numpy as np

def hit_rate_at_k(recommended_items: list[int], actual_items: set[int], k: int = 10) -> float:
    top_k = recommended_items[:k]
    for item in top_k:
        if item in actual_items:
            return 1.0
    return 0.0

def ndcg_at_k(recommended_items: list[int], actual_items: set[int], k: int = 10) -> float:
    top_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in actual_items:
            dcg += 1.0 / np.log2(i + 2)  # i+2 т.к. индексация с 0
            
    if dcg == 0.0:
        return 0.0
        
    # Идеальный DCG (если бы все релевантные были на верхних позициях)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k)))
    return dcg / idcg