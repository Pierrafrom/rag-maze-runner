# Image de l'application Streamlit RAG Maze Runner.
# Le LLM (Ollama) et l'index Chroma vivent HORS de l'image :
#   - Ollama : service séparé (voir docker-compose.yml) ;
#   - index  : monté en volume (jamais copié dans l'image).
FROM python:3.11-slim

# uv (gestionnaire de dépendances) depuis l'image officielle.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Dépendances système minimales (build de certaines wheels).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Couche dépendances (mise en cache tant que le manifeste ne change pas).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# 2) Code applicatif (change plus souvent → après les dépendances).
COPY src/ ./src/
COPY streamlit_app.py main.py ingest.py ./

# Streamlit : écoute sur toutes les interfaces, sans télémétrie ni prompt.
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "streamlit_app.py"]
