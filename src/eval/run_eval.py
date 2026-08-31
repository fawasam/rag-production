"""The Phase 3 evaluation run (FR-3.2): runs every golden_set.jsonl case through
the full Phase 2 pipeline, scores it with the four judge metrics in metrics.py,
and writes:
  - eval/eval_report.json   — full per-case detail for the latest run
  - eval/eval_history.jsonl — one summary row appended per run (trend data, FR-3.3)

This is also the script CI runs before a merge/deploy (FR-2.9/2.10) — see
gate.py, which reads eval_report.json and enforces the thresholds in SRS.md
section 8.

Run:
    python -m src.eval.run_eval
"""
import json
import time
from pathlib import Path

from src.eval.metrics import answer_relevance, context_precision, context_recall, faithfulness
from src.generation.grounded_client import generate_grounded_answer
from src.retrieval.hybrid import retrieve_hybrid

GOLDEN_SET_PATH = Path(__file__).resolve().parents[2] / "eval" / "golden_set.jsonl"
REPORT_PATH = Path(__file__).resolve().parents[2] / "eval" / "eval_report.json"
HISTORY_PATH = Path(__file__).resolve().parents[2] / "eval" / "eval_history.jsonl"


def load_golden_set() -> list[dict]:
    cases = []
    with open(GOLDEN_SET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_case(case: dict) -> dict:
    debug = retrieve_hybrid(case["question"], rerank_top_n=10, rerank_top_m=case.get("top_k", 5))
    result = generate_grounded_answer(case["question"], debug.reranked_results)
    contexts = [c.text for c in debug.reranked_results]
    retrieved_ids = [c.chunk_id for c in debug.reranked_results]

    reference = case.get("reference_answer", "")

    faithfulness_result = faithfulness(result.answer, contexts)
    relevance_result = answer_relevance(case["question"], result.answer)
    # context_precision/recall need a reference answer — skip gracefully if absent.
    if reference:
        precision_result = context_precision(case["question"], contexts, reference)
        recall_result = context_recall(contexts, reference)
    else:
        precision_result = {"score": None, "reasoning": "no reference_answer in golden set"}
        recall_result = {"score": None, "reasoning": "no reference_answer in golden set"}

    expected_ids = case.get("expected_chunk_ids") or (
        [case["expected_chunk_id"]] if case.get("expected_chunk_id") else []
    )
    retrieval_hit = all(eid in retrieved_ids for eid in expected_ids) if expected_ids else None

    return {
        "id": case["id"],
        "question": case["question"],
        "answer": result.answer,
        "answerable": result.answerable,
        "citations_valid": result.citations_valid,
        "retrieval_hit": retrieval_hit,
        "retrieved_chunk_ids": retrieved_ids,
        "metrics": {
            "faithfulness": faithfulness_result["score"],
            "context_precision": precision_result["score"],
            "context_recall": recall_result["score"],
            "answer_relevance": relevance_result["score"],
        },
        "metric_reasoning": {
            "faithfulness": faithfulness_result["reasoning"],
            "context_precision": precision_result["reasoning"],
            "context_recall": recall_result["reasoning"],
            "answer_relevance": relevance_result["reasoning"],
        },
    }


def _average(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def run_eval() -> dict:
    cases = load_golden_set()
    case_results = [run_case(case) for case in cases]

    aggregates = {
        "faithfulness": _average([r["metrics"]["faithfulness"] for r in case_results]),
        "context_precision": _average([r["metrics"]["context_precision"] for r in case_results]),
        "context_recall": _average([r["metrics"]["context_recall"] for r in case_results]),
        "answer_relevance": _average([r["metrics"]["answer_relevance"] for r in case_results]),
        "citation_validity_rate": _average(
            [1.0 if r["citations_valid"] else 0.0 for r in case_results]
        ),
        "retrieval_hit_rate": _average(
            [1.0 if r["retrieval_hit"] else 0.0 for r in case_results if r["retrieval_hit"] is not None]
        ),
        "num_cases": len(case_results),
    }

    report = {
        "timestamp": time.time(),
        "aggregates": aggregates,
        "cases": case_results,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))

    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps({"timestamp": report["timestamp"], **aggregates}) + "\n")

    return report


if __name__ == "__main__":
    report = run_eval()
    print(json.dumps(report["aggregates"], indent=2))
    print(f"\nFull report written to {REPORT_PATH}")
    print(f"History appended to {HISTORY_PATH}")
