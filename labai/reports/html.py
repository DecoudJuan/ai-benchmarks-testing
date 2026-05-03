"""
HTML report generator for Level 3 agent evaluation runs.

Produces a self-contained single-file HTML report with:
  - Run summary cards (overall, answer, reasoning, efficiency)
  - Per-category bar chart (inline SVG)
  - Per-item accordion: question, tool call trace (name + args + result), agent answer, scores
  - Fully responsive, dark-friendly design — no external dependencies
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime

from labai.core.types import EvalRecord, RunResult


# ── Score helpers ──────────────────────────────────────────────────────────────

def _score_class(score: float) -> str:
    if score >= 0.75:
        return "score-green"
    if score >= 0.50:
        return "score-yellow"
    return "score-red"


def _score_badge(score: float, label: str = "") -> str:
    cls   = _score_class(score)
    text  = f"{score:.0%}"
    inner = f"{label} {text}".strip()
    return f'<span class="badge {cls}">{html.escape(inner)}</span>'


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _h(text: str) -> str:
    return html.escape(str(text))


# ── SVG bar chart ──────────────────────────────────────────────────────────────

def _bar_chart_svg(cat_scores: dict[str, float], width: int = 560) -> str:
    if not cat_scores:
        return ""

    sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
    bar_h       = 32
    gap         = 8
    label_w     = 160
    bar_area    = width - label_w - 60
    total_h     = len(sorted_cats) * (bar_h + gap) + 20

    rows = []
    for i, (cat, score) in enumerate(sorted_cats):
        y     = i * (bar_h + gap) + 10
        bw    = max(2, int(bar_area * score))
        if score >= 0.75:
            colour = "#22c55e"
        elif score >= 0.50:
            colour = "#eab308"
        else:
            colour = "#ef4444"

        rows.append(
            f'<text x="{label_w - 8}" y="{y + bar_h // 2 + 5}" '
            f'text-anchor="end" class="chart-label">{_h(cat)}</text>'
            f'<rect x="{label_w}" y="{y}" width="{bw}" height="{bar_h}" '
            f'rx="4" fill="{colour}" opacity="0.85"/>'
            f'<text x="{label_w + bw + 6}" y="{y + bar_h // 2 + 5}" '
            f'class="chart-pct">{_h(_pct(score))}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {total_h}" '
        f'xmlns="http://www.w3.org/2000/svg" class="bar-chart">'
        + "\n".join(rows)
        + "</svg>"
    )


# ── Tool call trace ────────────────────────────────────────────────────────────

def _tool_trace_html(record: EvalRecord) -> str:
    calls = record.result.tool_calls
    if not calls:
        return '<p class="no-tools">No tool calls made.</p>'

    items = []
    for i, tc in enumerate(calls, 1):
        args_str = json.dumps(tc.arguments, indent=2)
        result_preview = tc.result[:600] + ("..." if len(tc.result) > 600 else "")
        items.append(
            f'<div class="tool-call">'
            f'  <div class="tool-header">'
            f'    <span class="tool-index">#{i}</span>'
            f'    <span class="tool-name">{_h(tc.name)}</span>'
            f'  </div>'
            f'  <div class="tool-body">'
            f'    <div class="tool-section-label">Arguments</div>'
            f'    <pre class="tool-args">{_h(args_str)}</pre>'
            f'    <div class="tool-section-label">Result</div>'
            f'    <pre class="tool-result">{_h(result_preview)}</pre>'
            f'  </div>'
            f'</div>'
        )
    return "\n".join(items)


# ── Per-item accordion row ─────────────────────────────────────────────────────

def _item_row_html(index: int, record: EvalRecord) -> str:
    item   = record.item
    result = record.result
    score  = record.score

    q_preview  = item.input.split("\n")[0][:90]
    n_tools    = len(result.tool_calls)
    tool_chain = " &rarr; ".join(f"<code>{_h(tc.name)}</code>" for tc in result.tool_calls) or "<em>none</em>"
    error_badge = '<span class="badge score-red">ERROR</span>' if result.error else ""
    rationale   = score.details.get("rationale", "")

    return f"""
<details class="item-row" id="item-{index}">
  <summary class="item-summary">
    <span class="item-idx">#{index + 1}</span>
    <span class="item-id">{_h(item.id)}</span>
    <span class="item-question">{_h(q_preview)}</span>
    <span class="item-tools-summary">{tool_chain}</span>
    <span class="item-scores">
      {_score_badge(score.overall,      "overall")}
      {_score_badge(score.answer_score, "answer")}
      {error_badge}
    </span>
  </summary>

  <div class="item-detail">

    <div class="detail-section">
      <h4>Question</h4>
      <pre class="question-text">{_h(item.input)}</pre>
    </div>

    <div class="detail-section">
      <h4>Tool Call Trace ({n_tools} call{'s' if n_tools != 1 else ''})</h4>
      {_tool_trace_html(record)}
    </div>

    <div class="detail-section">
      <h4>Agent Answer</h4>
      <pre class="answer-text">{_h(result.output or "(empty)")}</pre>
    </div>

    <div class="detail-section">
      <h4>Expected Answer</h4>
      <pre class="expected-text">{_h(item.expected)}</pre>
    </div>

    <div class="detail-section scores-detail">
      <h4>Scores</h4>
      <table class="score-table">
        <tr><th>Metric</th><th>Score</th><th>Weight</th></tr>
        <tr>
          <td>Answer correctness</td>
          <td>{_score_badge(score.answer_score)}</td>
          <td class="muted">60%</td>
        </tr>
        <tr>
          <td>Reasoning quality</td>
          <td>{_score_badge(score.reasoning_score)}</td>
          <td class="muted">30%</td>
        </tr>
        <tr>
          <td>Tool efficiency</td>
          <td>{_score_badge(score.efficiency_score)}</td>
          <td class="muted">10%</td>
        </tr>
        <tr class="overall-row">
          <td><strong>Overall</strong></td>
          <td>{_score_badge(score.overall)}</td>
          <td class="muted">weighted</td>
        </tr>
      </table>
      {'<p class="rationale"><em>' + _h(rationale) + '</em></p>' if rationale else ''}
    </div>

    {"<div class='detail-section error-section'><h4>Error</h4><pre>" + _h(result.error) + "</pre></div>" if result.error else ""}

    <div class="item-meta">
      {' '.join(f'<span class="meta-tag">{_h(k)}: {_h(str(v))}</span>' for k,v in item.metadata.items())}
      <span class="meta-tag">tokens: {result.total_tokens:,}</span>
      <span class="meta-tag">latency: {result.latency_ms:.0f}ms</span>
    </div>

  </div>
</details>
"""


# ── CSS ────────────────────────────────────────────────────────────────────────

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  line-height: 1.6;
  font-size: 14px;
}

a { color: #60a5fa; }

/* Layout */
.container { max-width: 1100px; margin: 0 auto; padding: 24px 16px 80px; }

/* Header */
.report-header {
  border-bottom: 1px solid #1e293b;
  padding-bottom: 20px;
  margin-bottom: 32px;
}
.report-header h1 { font-size: 1.6rem; font-weight: 700; color: #f8fafc; }
.report-header .subtitle { color: #94a3b8; font-size: 0.9rem; margin-top: 4px; }
.report-header .meta { display: flex; gap: 20px; margin-top: 12px; flex-wrap: wrap; }
.meta-item { font-size: 0.82rem; color: #64748b; }
.meta-item strong { color: #94a3b8; }

/* Summary cards */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 36px; }
.card {
  background: #1e293b;
  border-radius: 10px;
  padding: 18px 20px;
  border: 1px solid #334155;
}
.card .card-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: .06em; color: #64748b; margin-bottom: 6px; }
.card .card-value { font-size: 2rem; font-weight: 700; }
.card .card-sub   { font-size: 0.78rem; color: #64748b; margin-top: 2px; }
.card-green .card-value  { color: #22c55e; }
.card-yellow .card-value { color: #eab308; }
.card-red .card-value    { color: #ef4444; }
.card-blue .card-value   { color: #60a5fa; }

/* Section headers */
.section-title {
  font-size: 1rem;
  font-weight: 600;
  color: #cbd5e1;
  margin: 32px 0 14px;
  padding-bottom: 6px;
  border-bottom: 1px solid #1e293b;
  letter-spacing: .02em;
}

/* Chart */
.chart-wrap { background: #1e293b; border-radius: 10px; padding: 20px; border: 1px solid #334155; margin-bottom: 32px; }
.bar-chart { width: 100%; height: auto; display: block; }
.chart-label { font-size: 12px; fill: #94a3b8; font-family: inherit; }
.chart-pct   { font-size: 11px; fill: #64748b;  font-family: inherit; font-weight: 600; }

/* Badges */
.badge {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  white-space: nowrap;
}
.score-green  { background: #14532d; color: #4ade80; border: 1px solid #166534; }
.score-yellow { background: #422006; color: #fbbf24; border: 1px solid #78350f; }
.score-red    { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }

/* Item accordion */
.items-list { display: flex; flex-direction: column; gap: 6px; }

.item-row {
  background: #1e293b;
  border-radius: 8px;
  border: 1px solid #334155;
  overflow: hidden;
  transition: border-color .15s;
}
.item-row[open] { border-color: #475569; }
.item-row[open] .item-summary { background: #253347; }

.item-summary {
  display: grid;
  grid-template-columns: 32px 90px 1fr 180px auto;
  gap: 10px;
  align-items: center;
  padding: 10px 14px;
  cursor: pointer;
  list-style: none;
  user-select: none;
  transition: background .1s;
}
.item-summary:hover { background: #253347; }
.item-summary::-webkit-details-marker { display: none; }

.item-idx      { color: #475569; font-size: 0.78rem; font-weight: 600; }
.item-id       { font-family: monospace; font-size: 0.78rem; color: #7dd3fc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.item-question { font-size: 0.82rem; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.item-tools-summary { font-size: 0.75rem; color: #64748b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.item-scores   { display: flex; gap: 6px; justify-content: flex-end; flex-wrap: wrap; }

/* Item detail */
.item-detail { padding: 18px 20px; border-top: 1px solid #334155; display: flex; flex-direction: column; gap: 18px; }

.detail-section h4 { font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; color: #64748b; margin-bottom: 8px; }

pre {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 6px;
  padding: 12px 14px;
  font-size: 0.8rem;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  color: #cbd5e1;
  line-height: 1.5;
}
.question-text { max-height: 200px; overflow-y: auto; }
.answer-text   { max-height: 250px; overflow-y: auto; }
.expected-text { color: #86efac; }

/* Tool call trace */
.no-tools { color: #475569; font-size: 0.82rem; font-style: italic; }

.tool-call {
  border: 1px solid #1e3a5f;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 10px;
}
.tool-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  background: #0c2444;
  border-bottom: 1px solid #1e3a5f;
}
.tool-index { font-size: 0.72rem; background: #1e3a5f; color: #7dd3fc; padding: 1px 6px; border-radius: 4px; font-weight: 700; }
.tool-name  { font-family: monospace; font-size: 0.88rem; color: #38bdf8; font-weight: 600; }
.tool-body  { padding: 12px 14px; background: #0d1f38; display: flex; flex-direction: column; gap: 8px; }
.tool-section-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: .06em; color: #475569; }
.tool-args   { background: #0a1628; color: #93c5fd; border-color: #1e3a5f; }
.tool-result { background: #0a1628; color: #a7f3d0; border-color: #064e3b; max-height: 160px; overflow-y: auto; }

/* Score table */
.score-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.score-table th { text-align: left; padding: 6px 10px; color: #64748b; font-weight: 600; border-bottom: 1px solid #334155; }
.score-table td { padding: 7px 10px; border-bottom: 1px solid #1e293b; }
.score-table .overall-row td { border-top: 1px solid #334155; }
.muted { color: #475569; }

.rationale { font-size: 0.82rem; color: #94a3b8; margin-top: 10px; padding: 10px 12px; background: #0f172a; border-left: 3px solid #334155; border-radius: 0 4px 4px 0; }

.error-section pre { color: #fca5a5; border-color: #7f1d1d; background: #1c0505; }

/* Meta tags */
.item-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.meta-tag { font-size: 0.72rem; background: #0f172a; color: #64748b; border: 1px solid #1e293b; padding: 2px 8px; border-radius: 4px; }

/* Filter bar */
.filter-bar { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; align-items: center; }
.filter-bar label { font-size: 0.8rem; color: #64748b; }
.filter-input {
  background: #1e293b; border: 1px solid #334155; border-radius: 6px;
  color: #e2e8f0; padding: 5px 10px; font-size: 0.82rem; outline: none;
  flex: 1; min-width: 200px;
}
.filter-input:focus { border-color: #3b82f6; }
.filter-btn {
  background: #1e293b; border: 1px solid #334155; border-radius: 6px;
  color: #94a3b8; padding: 5px 12px; font-size: 0.8rem; cursor: pointer;
}
.filter-btn:hover { background: #334155; color: #e2e8f0; }
.filter-btn.active { background: #1d4ed8; border-color: #3b82f6; color: #fff; }

.items-hidden { display: none !important; }

/* Responsive */
@media (max-width: 700px) {
  .item-summary { grid-template-columns: 28px 1fr auto; }
  .item-id, .item-tools-summary { display: none; }
}
"""


# ── JS ─────────────────────────────────────────────────────────────────────────

_JS = """
function filterItems() {
  const q    = document.getElementById('search').value.toLowerCase();
  const filt = document.querySelector('.filter-btn.active')?.dataset.filter || 'all';
  document.querySelectorAll('.item-row').forEach(el => {
    const text  = el.textContent.toLowerCase();
    const score = parseFloat(el.dataset.overall || '0');
    const matchQ = !q || text.includes(q);
    const matchF = filt === 'all'
      || (filt === 'good'  && score >= 0.75)
      || (filt === 'mid'   && score >= 0.5 && score < 0.75)
      || (filt === 'bad'   && score < 0.5);
    el.classList.toggle('items-hidden', !(matchQ && matchF));
  });
}
document.getElementById('search').addEventListener('input', filterItems);
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    filterItems();
  });
});
"""


# ── Full page ──────────────────────────────────────────────────────────────────

def generate_agent_eval_html(run: RunResult, output_dir: str = "reports") -> str:
    """
    Generate a self-contained HTML report for an agent evaluation RunResult.

    Args:
        run:        The RunResult to report on.
        output_dir: Directory to save the HTML (created if needed).

    Returns:
        Absolute path to the saved HTML file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"agent_eval_{run.agent_name}_{run.run_id}.html"
    path     = os.path.join(output_dir, filename)

    # ── Aggregate values ───────────────────────────────────────────────────────
    def _card_class(score: float) -> str:
        if score >= 0.75: return "card-green"
        if score >= 0.50: return "card-yellow"
        return "card-red"

    overall_cls   = _card_class(run.avg_overall)
    answer_cls    = _card_class(run.avg_answer)
    reasoning_cls = _card_class(run.avg_reasoning)
    cat_scores    = run.scores_by_category()
    diff_scores   = run.scores_by_difficulty()

    # ── Summary cards ──────────────────────────────────────────────────────────
    cards_html = f"""
<div class="cards">
  <div class="card {overall_cls}">
    <div class="card-label">Overall Score</div>
    <div class="card-value">{_pct(run.avg_overall)}</div>
    <div class="card-sub">weighted avg</div>
  </div>
  <div class="card {answer_cls}">
    <div class="card-label">Answer Score</div>
    <div class="card-value">{_pct(run.avg_answer)}</div>
    <div class="card-sub">correctness</div>
  </div>
  <div class="card {reasoning_cls}">
    <div class="card-label">Reasoning</div>
    <div class="card-value">{_pct(run.avg_reasoning)}</div>
    <div class="card-sub">chain-of-thought</div>
  </div>
  <div class="card {_card_class(run.avg_efficiency)}">
    <div class="card-label">Efficiency</div>
    <div class="card-value">{_pct(run.avg_efficiency)}</div>
    <div class="card-sub">tool use</div>
  </div>
  <div class="card card-blue">
    <div class="card-label">Items</div>
    <div class="card-value">{len(run.records)}</div>
    <div class="card-sub">evaluated</div>
  </div>
  <div class="card card-blue">
    <div class="card-label">Total Tokens</div>
    <div class="card-value">{run.total_tokens:,}</div>
    <div class="card-sub">all calls</div>
  </div>
  <div class="card card-blue">
    <div class="card-label">Avg Tool Calls</div>
    <div class="card-value">{run.avg_tool_calls:.1f}</div>
    <div class="card-sub">per item</div>
  </div>
  <div class="card {'card-red' if run.error_rate > 0 else 'card-green'}">
    <div class="card-label">Error Rate</div>
    <div class="card-value">{_pct(run.error_rate)}</div>
    <div class="card-sub">failed items</div>
  </div>
</div>
"""

    # ── Charts ─────────────────────────────────────────────────────────────────
    charts_html = ""
    if cat_scores:
        charts_html += f"""
<div class="section-title">Score by Category</div>
<div class="chart-wrap">{_bar_chart_svg(cat_scores)}</div>
"""
    if diff_scores:
        charts_html += f"""
<div class="section-title">Score by Difficulty</div>
<div class="chart-wrap">{_bar_chart_svg(diff_scores, width=400)}</div>
"""

    # ── Item rows ──────────────────────────────────────────────────────────────
    items_html = "\n".join(
        f'<div data-overall="{rec.score.overall:.4f}">'
        + _item_row_html(i, rec)
        + "</div>"
        for i, rec in enumerate(run.records)
    )

    # ── Final HTML ─────────────────────────────────────────────────────────────
    now    = datetime.now().strftime("%Y-%m-%d %H:%M")
    page   = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LabAI Report - {_h(run.agent_name)} - {_h(run.run_id)}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="container">

  <div class="report-header">
    <h1>LabAI Agent Evaluation Report</h1>
    <div class="subtitle">Universidad Austral | AI Department</div>
    <div class="meta">
      <div class="meta-item"><strong>Agent:</strong> {_h(run.agent_name)}</div>
      <div class="meta-item"><strong>Dataset:</strong> {_h(run.dataset_name)}</div>
      <div class="meta-item"><strong>Run ID:</strong> {_h(run.run_id)}</div>
      <div class="meta-item"><strong>Date:</strong> {_h(now)}</div>
    </div>
  </div>

  {cards_html}

  {charts_html}

  <div class="section-title">Evaluation Results ({len(run.records)} items)</div>

  <div class="filter-bar">
    <label>Filter:</label>
    <input id="search" class="filter-input" type="text" placeholder="Search questions, tools, answers...">
    <button class="filter-btn active" data-filter="all">All</button>
    <button class="filter-btn" data-filter="good">Good (&ge;75%)</button>
    <button class="filter-btn" data-filter="mid">Mid (50-75%)</button>
    <button class="filter-btn" data-filter="bad">Low (&lt;50%)</button>
  </div>

  <div class="items-list">
    {items_html}
  </div>

</div>
<script>{_JS}</script>
</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(page)

    return os.path.abspath(path)
