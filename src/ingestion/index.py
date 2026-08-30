"""Embed chunks with OpenAI and load them into a local Chroma collection.

Run directly to (re)build the index from data/raw/:
    python -m src.ingestion.index
"""
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from src.ingestion.chunk import chunk_all
from src.retrieval.bm25 import build_bm25_index

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "chroma"
COLLECTION_NAME = "rag_chunks"
EMBED_BATCH_SIZE = 100


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


def build_index(raw_dir: Path = None) -> int:
    raw_dir = raw_dir or (Path(__file__).resolve().parents[2] / "data" / "raw")
    chunks = chunk_all(raw_dir)
    if not chunks:
        raise RuntimeError(f"No .md/.txt documents found in {raw_dir}")

    client = get_openai_client()
    texts = [c.text for c in chunks]
    embeddings = embed_texts(client, texts)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Fresh build each run for Phase 1 simplicity.
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {"doc_id": c.doc_id, "source_path": c.source_path, "position": c.position}
            for c in chunks
        ],
    )

    # Sparse (BM25) index — the other half of Phase 2 hybrid search.
    build_bm25_index(chunks)

    return len(chunks)


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks into {CHROMA_DIR} (dense) and BM25 (sparse)")
