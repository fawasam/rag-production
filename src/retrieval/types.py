"""Shared retrieval types, used by dense.py, bm25.py, rrf.py, and rerank/."""
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_id: str
    source_path: str
    score: float  # meaning depends on stage: similarity (dense), BM25 score,
    # RRF score, or cross-encoder relevance score. Not comparable across stages.
