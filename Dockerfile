# 1. Используем официальный бинарник uv
FROM ghcr.io/astral-sh/uv:latest AS uv

# 2. Базовый легковесный образ Python 3.12
FROM python:3.12-slim

WORKDIR /app

# Копируем uv из первого слоя
COPY --from=uv /uv /uvx /bin/

# Копируем файлы зависимостей
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости с кэшированием (без самого проекта)
RUN uv sync --frozen --no-install-project

# Копируем код, обученные модели и обработанные фичи
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Пробрасываем порт FastAPI
EXPOSE 8000

# Запускаем uvicorn внутри uv окружения
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]