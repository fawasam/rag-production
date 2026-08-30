"""Cross-encoder re-ranking: score each (query, chunk) pair jointly for a more
accurate relevance signal than the fused RRF rank alone (FR-2.3/FR-2.4).

A cross-encoder is slower per-pair than dense/BM25 (it can't be precomputed —
every query needs a fresh forward pass per candidate), which is exactly why it
only runs on the top-N *fused* candidates, not the whole corpus.
"""
from sentence_transformers import CrossEncoder

from src.retrieval.types import RetrievedChunk

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[RetrievedChunk], top_m: int = 5) -> list[RetrievedChunk]:
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    rescored = [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            text=c.text,
            doc_id=c.doc_id,
            source_path=c.source_path,
            score=float(score),
        )
        for c, score in zip(candidates, scores)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored[:top_m]
