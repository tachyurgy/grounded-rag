FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    DB_PATH=/data/grounded.db

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY web ./web
COPY data/corpus ./data/corpus

RUN mkdir -p /data
EXPOSE 8000

# Non-root runtime
RUN useradd -m runner && chown -R runner:runner /app /data
USER runner

# kamal-proxy health check hits /up
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
