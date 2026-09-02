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
    security_debug: dict | None = None,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": timestamp,
        "query": query,
        "latency_seconds": round(latency_seconds, 3),
        "security_debug": security_debug or {"pii_detected": 0, "injection_blocked": False},
        "retrieval_debug": {
            "dense": [c.chunk_id for c in debug.dense_results] if debug else [],
            "bm25": [c.chunk_id for c in debug.bm25_results] if debug else [],
            "fused": [c.chunk_id for c in debug.fused_results] if debug else [],
            "reranked": [c.chunk_id for c in debug.reranked_results] if debug else [],
        },
        "answer": result.answer if result else "",
        "answerable": result.answerable if result else False,
        "citations": result.citations if result else [],
        "citations_valid": result.citations_valid if result else False,
        "invalid_citations": [vars(ic) for ic in result.invalid_citations] if result else [],
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
