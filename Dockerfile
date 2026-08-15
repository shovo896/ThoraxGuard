FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app.py ./
COPY config ./config
COPY params.yaml ./
COPY templates ./templates
COPY static ./static
COPY artifacts/training/model.keras ./artifacts/training/model.keras

EXPOSE 8080

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 180 app:app"]




