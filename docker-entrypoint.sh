#!/bin/sh
# On first run (empty mounted volume), build the dense+sparse index before
# serving. On later runs it's a fast incremental no-op (FR-2.8) unless
# data/raw has actually changed.
set -e

if [ ! -d "/app/data/processed/chroma" ]; then
  echo "No index found in the mounted volume — running initial ingestion..."
fi
python -m src.ingestion.index

exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
