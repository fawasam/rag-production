"""Trend dashboard (FR-3.3): renders eval/eval_history.jsonl as a self-contained
static HTML page — one line chart per metric across all runs, plus a
latest-run-vs-threshold table. No external services, no JS framework; open
the file directly in a browser.

This is a local project artifact, not a live/hosted dashboard — swapping in
Grafana+Prometheus (per SRS.md Appendix B) is the natural next step once
there's real production traffic to watch, rather than golden-set replay runs.

Run:
    python -m src.eval.report_html
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from src.eval.gate import THRESHOLDS

HISTORY_PATH = Path(__file__).resolve().parents[2] / "eval" / "eval_history.jsonl"
DASHBOARD_PATH = Path(__file__).resolve().parents[2] / "eval" / "dashboard.html"

METRIC_COLORS = {
    "faithfulness": "#2563eb",
    "context_precision": "#dc2626",
    "context_recall": "#16a34a",
    "answer_relevance": "#9333ea",
    "citation_validity_rate": "#ea580c",
    "retrieval_hit_rate": "#0891b2",
}

CHART_W, CHART_H = 640, 220
PAD_L, PAD_R, PAD_T, PAD_B = 45, 15, 15, 25


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def _svg_line_chart(metric: str, history: list[dict]) -> str:
    values = [run.get(metric) for run in history]
    points = [(i, v) for i, v in enumerate(values) if v is not None]
    color = METRIC_COLORS.get(metric, "#333")

    if len(points) < 2:
        # Not enough runs yet for a trend line — show the single point (or
        # nothing) rather than a misleading/empty chart.
        note = "not enough runs yet for a trend line" if points else "no data"
        return f'<div class="chart-empty">{metric}: {note}</div>'

    n = len(values)
    x_scale = (CHART_W - PAD_L - PAD_R) / max(n - 1, 1)
    y_scale = CHART_H - PAD_T - PAD_B

    def to_svg_xy(i, v):
        x = PAD_L + i * x_scale
        y = PAD_T + (1 - v) * y_scale
        return x, y

    path = " ".join(
        f"{'M' if idx == 0 else 'L'}{x:.1f},{y:.1f}"
        for idx, (x, y) in enumerate(to_svg_xy(i, v) for i, v in points)
    )
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'
        for x, y in (to_svg_xy(i, v) for i, v in points)
    )
    gridlines = "".join(
        f'<line x1="{PAD_L}" y1="{PAD_T + f * y_scale:.1f}" '
        f'x2="{CHART_W - PAD_R}" y2="{PAD_T + f * y_scale:.1f}" '
        f'stroke="var(--grid)" stroke-width="1"/>'
        f'<text x="4" y="{PAD_T + f * y_scale + 4:.1f}" font-size="10" fill="var(--muted)">{1 - f:.1f}</text>'
        for f in (0.0, 0.5, 1.0)
    )

    return f"""
    <div class="chart">
      <div class="chart-title">{metric}</div>
      <svg viewBox="0 0 {CHART_W} {CHART_H}" width="100%" height="{CHART_H}">
        {gridlines}
        <path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>
        {dots}
      </svg>
    </div>
    """


def _threshold_table(latest: dict) -> str:
    rows = []
    for metric, threshold, is_hard in THRESHOLDS:
        value = latest.get(metric)
        if value is None:
            status, value_str = "no data", "—"
        else:
            ok = value >= threshold
            status = "PASS" if ok else "FAIL"
            value_str = f"{value:.3f}"
        gate = "hard" if is_hard else "soft"
        row_class = "pass" if status == "PASS" else ("fail" if status == "FAIL" else "nodata")
        rows.append(
            f"<tr class='{row_class}'><td>{metric}</td><td>{value_str}</td>"
            f"<td>{threshold}</td><td>{gate}</td><td>{status}</td></tr>"
        )
    return "\n".join(rows)


def render_dashboard() -> str:
    history = load_history()
    latest = history[-1] if history else {}
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    charts = "\n".join(
        _svg_line_chart(metric, history)
        for metric in [
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_relevance",
            "citation_validity_rate",
            "retrieval_hit_rate",
        ]
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>RAG Eval Dashboard</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --grid: #e5e7eb;
    --card: #f9fafb; --border: #e5e7eb;
    --pass: #16a34a; --fail: #dc2626;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #0f172a; --fg: #e2e8f0; --muted: #94a3b8; --grid: #1e293b;
              --card: #1e293b; --border: #334155; }}
  }}
  body {{ background: var(--bg); color: var(--fg); font-family: -apple-system, sans-serif;
          max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 2rem; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }}
  tr.pass td:last-child {{ color: var(--pass); font-weight: 600; }}
  tr.fail td:last-child {{ color: var(--fail); font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  .chart {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }}
  .chart-title {{ font-size: 0.85rem; font-weight: 600; margin-bottom: 4px; }}
  .chart-empty {{ background: var(--card); border: 1px dashed var(--border); border-radius: 8px;
                   padding: 20px; color: var(--muted); font-size: 0.85rem; text-align: center; }}
</style>
</head>
<body>
  <h1>RAG Evaluation Dashboard</h1>
  <div class="meta">Generated {generated_at} · {len(history)} run(s) in history</div>

  <h2>Latest run vs. thresholds (SRS.md section 8)</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Gate</th><th>Status</th></tr>
    {_threshold_table(latest)}
  </table>

  <h2>Trends over time</h2>
  <div class="grid">
    {charts}
  </div>
</body>
</html>
"""


if __name__ == "__main__":
    DASHBOARD_PATH.write_text(render_dashboard())
    print(f"Dashboard written to {DASHBOARD_PATH}")
