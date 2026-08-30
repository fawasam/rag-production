"""BM25 sparse retrieval — the keyword-matching half of Phase 2 hybrid search.

The index is built once during ingestion (see ingestion/index.py) and persisted
as a pickle; query time just tokenizes the query and scores it against the
persisted corpus. Good enough for a corpus small enough to fit in memory —
migrate to OpenSearch/Elasticsearch (FR-2.1) if that stops being true.
"""
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.retrieval.types import RetrievedChunk

BM25_INDEX_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "bm25_index.pkl"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization. Simple on purpose — BM25's value is
    exact/near-exact term matching, not linguistic sophistication."""
    return _TOKEN_RE.findall(text.lower())


def build_bm25_index(chunks) -> None:
    """chunks: list[Chunk] from ingestion.chunk.chunk_all()."""
    tokenized_corpus = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "bm25": bm25,
        "chunk_ids": [c.chunk_id for c in chunks],
        "doc_ids": [c.doc_id for c in chunks],
        "source_paths": [c.source_path for c in chunks],
        "texts": [c.text for c in chunks],
    }
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)


_cache = None


def _load_index():
    global _cache
    if _cache is None:
        if not BM25_INDEX_PATH.exists():
            raise RuntimeError(
                f"BM25 index not found at {BM25_INDEX_PATH}. "
                "Run `python -m src.ingestion.index` first."
            )
        with open(BM25_INDEX_PATH, "rb") as f:
            _cache = pickle.load(f)
    return _cache


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    index = _load_index()
    bm25: BM25Okapi = index["bm25"]
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        RetrievedChunk(
            chunk_id=index["chunk_ids"][i],
            text=index["texts"][i],
            doc_id=index["doc_ids"][i],
            source_path=index["source_paths"][i],
            score=float(scores[i]),
        )
        for i in ranked
        if scores[i] > 0  # exclude chunks with zero term overlap
    ]


if __name__ == "__main__":
    for r in retrieve("API rate limit 429", top_k=3):
        print(f"[{r.score:.3f}] {r.chunk_id}\n{r.text[:200]}...\n")
