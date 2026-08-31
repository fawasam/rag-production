"""Rolling-window alerting (FR-3.4): checks the last N eval runs' hard-gated
metrics, not just the latest single run — a real regression looks different
from one noisy run.

This prints a structured alert to stdout/log. It does NOT deliver to
Slack/PagerDuty — those require the corresponding MCP connectors to be
authorized for this project, which they are not in this environment. Wiring
delivery is a small, isolated change once that's set up: replace `_emit_alert`
below with an actual Slack/PagerDuty API call.

Run:
    python -m src.eval.alerts
"""
import json
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parents[2] / "eval" / "eval_history.jsonl"
ROLLING_WINDOW = 3

# Same hard gates as gate.py — alerting watches the metrics that actually block deploys.
HARD_GATE_THRESHOLDS = {
    "faithfulness": 0.85,
    "citation_validity_rate": 1.00,
}


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_rolling_window(history: list[dict], window: int = ROLLING_WINDOW) -> list[str]:
    """Returns a list of alert messages (empty if nothing to alert on)."""
    if not history:
        return []

    recent = history[-window:]
    alerts = []

    for metric, threshold in HARD_GATE_THRESHOLDS.items():
        values = [run.get(metric) for run in recent if run.get(metric) is not None]
        if not values:
            continue
        rolling_avg = sum(values) / len(values)
        if rolling_avg < threshold:
            alerts.append(
                f"ALERT: {metric} rolling average over last {len(values)} run(s) is "
                f"{rolling_avg:.3f}, below hard-gate threshold {threshold}. "
                f"Recent values: {[round(v, 3) for v in values]}"
            )
    return alerts


def _emit_alert(message: str) -> None:
    # Placeholder delivery — see module docstring. Swap this for a real
    # Slack/PagerDuty call once those connectors are authorized.
    print(f"🚨 {message}")


def main() -> None:
    history = load_history()
    if not history:
        print("No eval history yet — run `python -m src.eval.run_eval` first.")
        return

    alerts = check_rolling_window(history)
    if not alerts:
        print(f"No alerts. Rolling window of last {min(ROLLING_WINDOW, len(history))} run(s) OK.")
        return

    for alert in alerts:
        _emit_alert(alert)


if __name__ == "__main__":
    main()
