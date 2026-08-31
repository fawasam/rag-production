"""Query logging (FR-3.1): every production query is logged as one JSON line
with retrieved chunk IDs, the full retrieval debug trace, the final answer,
citations, citation validity, and latency. This is what feeds Phase 3
faithfulness sampling (FR-3.2) and any future dashboard/alerting.
"""
import json
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "logs" / "queries.jsonl"


def log_query(
    query: str,
    debug,  # HybridRetrievalDebug
    result,  # GroundedAnswer
    latency_seconds: float,
    timestamp: float,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": timestamp,
        "query": query,
        "latency_seconds": round(latency_seconds, 3),
        "retrieval_debug": {
            "dense": [c.chunk_id for c in debug.dense_results],
            "bm25": [c.chunk_id for c in debug.bm25_results],
            "fused": [c.chunk_id for c in debug.fused_results],
            "reranked": [c.chunk_id for c in debug.reranked_results],
        },
        "answer": result.answer,
        "answerable": result.answerable,
        "citations": result.citations,
        "citations_valid": result.citations_valid,
        "invalid_citations": [vars(ic) for ic in result.invalid_citations],
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


class Timer:
    """Small context manager so callers don't have to juggle time.time() calls.

    Usage:
        with Timer() as t:
            ... do work ...
        log_query(..., latency_seconds=t.elapsed, timestamp=t.start)
    """

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.start
