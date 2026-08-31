"""CI/CD evaluation gate (FR-2.9/FR-2.10): reads the latest eval_report.json
and enforces the thresholds from SRS.md section 8. Exits non-zero (fails the
build) if any HARD-gated metric misses threshold.

Run:
    python -m src.eval.run_eval   # produces eval/eval_report.json
    python -m src.eval.gate       # checks it against thresholds
"""
import json
import sys
from pathlib import Path

REPORT_PATH = Path(__file__).resolve().parents[2] / "eval" / "eval_report.json"

# (metric, minimum threshold, is_hard_gate) — from SRS.md section 8.
THRESHOLDS = [
    ("faithfulness", 0.85, True),
    ("context_precision", 0.75, False),
    ("context_recall", 0.80, False),
    ("answer_relevance", 0.85, False),
    ("citation_validity_rate", 1.00, True),  # 100% hard gate, per SRS
]


def check_gate(aggregates: dict) -> tuple[bool, list[str]]:
    """Returns (passed, messages). passed=False if any HARD gate is violated."""
    passed = True
    messages = []
    for metric, threshold, is_hard in THRESHOLDS:
        value = aggregates.get(metric)
        if value is None:
            messages.append(f"  ? {metric}: no data (skipped)")
            continue
        ok = value >= threshold
        gate_label = "HARD GATE" if is_hard else "soft"
        status = "PASS" if ok else "FAIL"
        messages.append(f"  [{status}] {metric}: {value:.3f} (threshold {threshold}, {gate_label})")
        if not ok and is_hard:
            passed = False
    return passed, messages


def main() -> int:
    if not REPORT_PATH.exists():
        print(f"No eval report found at {REPORT_PATH}. Run `python -m src.eval.run_eval` first.")
        return 1

    report = json.loads(REPORT_PATH.read_text())
    aggregates = report["aggregates"]

    print(f"Evaluation gate — {aggregates['num_cases']} cases")
    passed, messages = check_gate(aggregates)
    print("\n".join(messages))

    if passed:
        print("\n✅ Gate PASSED — hard-gated metrics all meet threshold.")
        return 0
    else:
        print("\n❌ Gate FAILED — a hard-gated metric is below threshold. Blocking deploy.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
