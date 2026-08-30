"""Phase 2 retrieval pipeline: dense + BM25 -> RRF fusion -> cross-encoder rerank.

This is what FR-2.1 through FR-2.4 describe end to end. Each stage is also
independently importable/testable (dense.py, bm25.py, rrf.py, rerank/cross_encoder.py).
"""
from dataclasses import dataclass

from src import config
from src.rerank.cross_encoder import rerank
from src.retrieval import bm25, dense
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.types import RetrievedChunk


@dataclass
class HybridRetrievalDebug:
    """Full trace of every stage — useful for debugging and for the eval
    harness (e.g. checking whether a chunk was lost at fusion vs. rerank)."""

    dense_results: list[RetrievedChunk]
    bm25_results: list[RetrievedChunk]
    fused_results: list[RetrievedChunk]
    reranked_results: list[RetrievedChunk]


def retrieve_hybrid(
    query: str,
    dense_top_k: int = config.DENSE_TOP_K,
    bm25_top_k: int = config.BM25_TOP_K,
    rrf_k: int = config.RRF_K,
    rerank_top_n: int = config.RERANK_TOP_N,
    rerank_top_m: int = config.RERANK_TOP_M,
) -> HybridRetrievalDebug:
    dense_results = dense.retrieve(query, top_k=dense_top_k)
    bm25_results = bm25.retrieve(query, top_k=bm25_top_k)

    fused = reciprocal_rank_fusion([dense_results, bm25_results], k=rrf_k)
    fused_top_n = fused[:rerank_top_n]

    reranked = rerank(query, fused_top_n, top_m=rerank_top_m)

    return HybridRetrievalDebug(
        dense_results=dense_results,
        bm25_results=bm25_results,
        fused_results=fused,
        reranked_results=reranked,
    )


if __name__ == "__main__":
    debug = retrieve_hybrid("What is the API rate limit for Enterprise plans?")
    print("Dense:", [c.chunk_id for c in debug.dense_results])
    print("BM25:", [c.chunk_id for c in debug.bm25_results])
    print("Fused:", [c.chunk_id for c in debug.fused_results])
    print("Reranked (final):", [c.chunk_id for c in debug.reranked_results])
