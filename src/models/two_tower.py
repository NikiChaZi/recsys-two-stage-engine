import torch
import torch.nn as nn
import torch.nn.functional as F

class Tower(nn.Module):
    def __init__(self, num_entities: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, embedding_dim)
        
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # <-- LayerNorm вместо BatchNorm1d
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids)
        x = self.projector(x)
        return F.normalize(x, p=2, dim=1)

class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.user_tower = Tower(num_users, embedding_dim)
        self.item_tower = Tower(num_movies, embedding_dim)
        self.temperature = temperature

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_tower(user_ids)    # (B, embedding_dim)
        item_emb = self.item_tower(movie_ids)   # (B, embedding_dim)
        return user_emb, item_emb

    def compute_loss(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        similarity = torch.matmul(user_emb, item_emb.T) / self.temperature
        targets = torch.arange(user_emb.size(0), device=user_emb.device)
        
        # Симметричный контрастивный лосс (User-to-Item + Item-to-User)
        loss_u2i = F.cross_entropy(similarity, targets)
        loss_i2u = F.cross_entropy(similarity.T, targets)
        return (loss_u2i + loss_i2u) / 2.0


if __name__ == "__main__":
    # Быстрый тест на фиктивном батче
    batch_size = 1024
    num_users = 6037
    num_movies = 3456

    model = TwoTowerModel(num_users, num_movies, embedding_dim=64)
    
    dummy_users = torch.randint(0, num_users, (batch_size,))
    dummy_movies = torch.randint(0, num_movies, (batch_size,))
    
    u_emb, i_emb = model(dummy_users, dummy_movies)
    loss = model.compute_loss(u_emb, i_emb)
    
    print(f"Форма user embeddings: {u_emb.shape}")
    print(f"Форма item embeddings: {i_emb.shape}")
    print(f"Стартовый InfoNCE Loss: {loss.item():.4f}")
    
    # Теоретический лосс для случайных предсказаний равен ln(B)
    import math
    print(f"Теоретический случайный лосс: ln({batch_size}) = {math.log(batch_size):.4f}")