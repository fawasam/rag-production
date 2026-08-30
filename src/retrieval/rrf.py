"""Reciprocal Rank Fusion: merge ranked lists from dense + BM25 retrievers.

RRF ignores the raw scores from each retriever (which live on incomparable
scales — cosine similarity vs. BM25 term-frequency scores) and fuses purely
on rank position:

    rrf_score(chunk) = sum over retrievers where chunk appears of  1 / (k + rank)

A chunk that ranks well in both lists beats one that ranks #1 in only one —
that's the point: it rewards agreement between keyword and semantic search.
"""
from src.retrieval.types import RetrievedChunk

DEFAULT_RRF_K = 60  # standard constant from the original RRF paper (Cormack et al., 2009)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k: int = DEFAULT_RRF_K
) -> list[RetrievedChunk]:
    """ranked_lists: e.g. [dense_results, bm25_results], each already sorted
    best-first. Returns a single list, sorted by fused RRF score, descending.
    """
    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                k + rank
            )
            # Keep the first-seen chunk object for its text/metadata (identical
            # across retrievers for a given chunk_id).
            chunk_lookup.setdefault(chunk.chunk_id, chunk)

    fused = [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=chunk_lookup[chunk_id].text,
            doc_id=chunk_lookup[chunk_id].doc_id,
            source_path=chunk_lookup[chunk_id].source_path,
            score=score,
        )
        for chunk_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused
