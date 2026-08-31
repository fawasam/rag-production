# Production image for the RAG API. Python 3.11 (not the host's 3.14) for
# broad prebuilt-wheel availability — torch/sentence-transformers in
# particular need mature wheel support.
FROM python:3.11-slim

WORKDIR /app

# hnswlib (a chromadb dependency) needs a C++ compiler to build from source
# on some platforms/wheel-less architectures.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/raw/ ./data/raw/
COPY eval/golden_set.jsonl ./eval/golden_set.jsonl
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Run as a non-root user. Create data/processed and data/logs here (rather
# than relying on Docker to create them at mount time) so their ownership is
# already correct before the volume takes over the directory's contents.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/processed /app/data/logs \
    && chown -R appuser:appuser /app
USER appuser

# data/processed (indices) and data/logs (query logs) are meant to be mounted
# volumes — they're generated/written at runtime, not baked into the image.
VOLUME ["/app/data/processed", "/app/data/logs"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
