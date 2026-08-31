"""Embed chunks with OpenAI and load them into a local Chroma collection.

Incremental by default (FR-2.8): skips re-embedding documents whose mtime
hasn't changed since the last successful ingest (tracked in
ingestion_manifest.json). Re-embedding is the expensive, API-billed step —
a full BM25 rebuild every run is cheap and local, so that always happens
regardless.

Run:
    python -m src.ingestion.index          # incremental (default)
    python -m src.ingestion.index --full   # force a full rebuild
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from src.ingestion.chunk import chunk_all
from src.ingestion.manifest import load_manifest, save_manifest
from src.ingestion.parsers import SUPPORTED_EXTENSIONS
from src.retrieval.bm25 import build_bm25_index

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "chroma"
COLLECTION_NAME = "rag_chunks"
EMBED_BATCH_SIZE = 100
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend([item.embedding for item in resp.data])
    return embeddings


def _scan_doc_mtimes(raw_dir: Path) -> dict[str, float]:
    return {
        path.name: path.stat().st_mtime
        for path in sorted(raw_dir.glob("**/*"))
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file()
    }


def build_index(raw_dir: Path = None, force_full: bool = False) -> dict:
    raw_dir = raw_dir or DEFAULT_RAW_DIR
    chunks = chunk_all(raw_dir)
    if not chunks:
        raise RuntimeError(f"No supported documents found in {raw_dir}")

    chunks_by_doc: dict[str, list] = defaultdict(list)
    for c in chunks:
        chunks_by_doc[c.doc_id].append(c)
    current_mtimes = _scan_doc_mtimes(raw_dir)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    manifest = {} if force_full else load_manifest()
    deleted_docs: set[str] = set()

    if force_full or not manifest:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = chroma_client.create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        docs_to_embed = set(chunks_by_doc.keys())
        manifest = {}
    else:
        collection = chroma_client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        docs_to_embed = {
            doc_id
            for doc_id in chunks_by_doc
            if doc_id not in manifest or manifest[doc_id]["mtime"] != current_mtimes[doc_id]
        }
        deleted_docs = set(manifest.keys()) - set(chunks_by_doc.keys())

        for doc_id in deleted_docs:
            collection.delete(ids=manifest[doc_id]["chunk_ids"])
            del manifest[doc_id]

        for doc_id in docs_to_embed:
            if doc_id in manifest:
                # Doc changed — chunk boundaries may have shifted, so drop
                # its old chunks before re-adding the freshly computed ones.
                collection.delete(ids=manifest[doc_id]["chunk_ids"])

    if docs_to_embed:
        client = get_openai_client()
        chunks_to_embed = [c for doc_id in docs_to_embed for c in chunks_by_doc[doc_id]]
        texts = [c.text for c in chunks_to_embed]
        embeddings = embed_texts(client, texts)

        collection.add(
            ids=[c.chunk_id for c in chunks_to_embed],
            embeddings=embeddings,
            documents=[c.text for c in chunks_to_embed],
            metadatas=[
                {"doc_id": c.doc_id, "source_path": c.source_path, "position": c.position}
                for c in chunks_to_embed
            ],
        )
        for doc_id in docs_to_embed:
            manifest[doc_id] = {
                "mtime": current_mtimes[doc_id],
                "chunk_ids": [c.chunk_id for c in chunks_by_doc[doc_id]],
            }

    save_manifest(manifest)

    # Sparse (BM25) index: always a full rebuild. It's cheap, local, needs the
    # whole corpus to compute IDF correctly, and costs no API calls either way.
    build_bm25_index(chunks)

    return {
        "total_chunks": len(chunks),
        "docs_embedded": sorted(docs_to_embed),
        "docs_deleted": sorted(deleted_docs),
        "docs_unchanged": sorted(set(chunks_by_doc.keys()) - docs_to_embed),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true", help="Force a full rebuild, ignoring the manifest."
    )
    args = parser.parse_args()

    result = build_index(force_full=args.full)
    print(f"Total chunks: {result['total_chunks']}")
    print(f"Embedded ({len(result['docs_embedded'])}): {result['docs_embedded']}")
    print(f"Unchanged, skipped ({len(result['docs_unchanged'])}): {result['docs_unchanged']}")
    if result["docs_deleted"]:
        print(f"Removed ({len(result['docs_deleted'])}): {result['docs_deleted']}")
    sys.stdout.flush()
