import torch
from pathlib import Path
from tqdm import tqdm

from src.data.dataset import prepare_data_and_mappings
from src.models.two_tower import TwoTowerModel

def train(
    epochs: int = 15,       # 15 эпох вместо 7
    batch_size: int = 512,  # 512 вместо 1024 (меньше ложных негативов)
    lr: float = 5e-3,       # 5e-3 вместо 1e-3 (быстрее сходимость)
    embedding_dim: int = 64,
):

    # 1. Выбираем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}\n")

    # 2. Подготовка данных
    dataloader, mappings = prepare_data_and_mappings(batch_size=batch_size)
    num_users = len(mappings["user2idx"])
    num_movies = len(mappings["movie2idx"])

    # 3. Инициализация модели
    model = TwoTowerModel(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=embedding_dim,
        temperature=0.07,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 4. Цикл обучения
    print("\nНачинаем обучение:")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for users, movies in pbar:
            users = users.to(device)
            movies = movies.to(device)

            optimizer.zero_grad()
            user_emb, item_emb = model(users, movies)
            loss = model.compute_loss(user_emb, item_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} закончена | Средний InfoNCE Loss: {avg_loss:.4f}")

    # 5. Сохранение весов и маппингов
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), models_dir / "two_tower.pt")
    torch.save(mappings, models_dir / "mappings.pt")
    
    print(f"\nМодель и маппинги успешно сохранены в папку {models_dir.resolve()}!")

if __name__ == "__main__":
    train()