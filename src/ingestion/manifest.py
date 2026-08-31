"""Ingestion manifest (FR-2.8): tracks each source doc's mtime and the chunk_ids
it produced, so re-ingestion can skip re-embedding unchanged documents — the
expensive, API-billed step — instead of doing a full rebuild every run.
"""
import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "ingestion_manifest.json"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
