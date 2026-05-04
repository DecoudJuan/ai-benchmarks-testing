"""
HTML report generator for Level 3 agent evaluation runs.

Accepts one or multiple RunResult objects and produces a single self-contained HTML file:
  - Single model  → summary cards + CSS bar charts + per-item accordion
  - Multi model   → tabbed layout: one tab per model + "Compare" tab
"""

from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from datetime import datetime

from labai.core.types import EvalRecord, RunResult

# ── Helpers ────────────────────────────────────────────────────────────────────

def _h(text: str) -> str:
    return html.escape(str(text))

def _pct(v: float) -> str:
    return f"{v:.1%}"

def _score_cls(score: float) -> str:
    if score >= 0.75: return "green"
    if score >= 0.50: return "yellow"
    return "red"

def _badge(score: float, label: str = "") -> str:
    c = _score_cls(score)
    t = f"{label} {score:.0%}".strip()
    return f'<span class="badge {c}">{_h(t)}</span>'

# ── CSS bar chart (percentage-based, responsive) ───────────────────────────────

def _bar_chart(scores: dict[str, float], colors: list[str] | None = None) -> str:
    """Render a compact CSS bar chart. scores = {label: 0-1}."""
    if not scores:
        return ""
    rows = []
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (label, score) in enumerate(sorted_items):
        c = _score_cls(score)
        color_override = f'style="background:{colors[i % len(colors)]}"' if colors else ""
        rows.append(f"""
      <div class="bar-row">
        <span class="bar-label">{_h(label)}</span>
        <div class="bar-track">
          <div class="bar-fill {c}" style="width:{score*100:.1f}%" {color_override}></div>
        </div>
        <span class="bar-pct">{score*100:.1f}%</span>
      </div>""")
    return f'<div class="bar-chart">{"".join(rows)}\n</div>'


def _multi_bar_chart(scores_per_model: dict[str, dict[str, float]], model_colors: dict[str, str]) -> str:
    """For each category, show one bar per model side by side."""
    if not scores_per_model:
        return ""
    # Collect all categories
    all_cats: set[str] = set()
    for d in scores_per_model.values():
        all_cats.update(d.keys())
    all_cats_sorted = sorted(all_cats)

    rows = []
    for cat in all_cats_sorted:
        model_bars = []
        for model, color in model_colors.items():
            score = scores_per_model.get(model, {}).get(cat, 0.0)
            model_bars.append(f"""
          <div class="multi-bar-row">
            <span class="multi-bar-model" style="color:{color}">{_h(model)}</span>
            <div class="bar-track">
              <div class="bar-fill" style="width:{score*100:.1f}%;background:{color};opacity:0.85"></div>
            </div>
            <span class="bar-pct">{score*100:.1f}%</span>
          </div>""")
        rows.append(f"""
      <div class="multi-bar-group">
        <div class="multi-bar-cat">{_h(cat)}</div>
        {"".join(model_bars)}
      </div>""")
    return f'<div class="multi-bar-chart">{"".join(rows)}\n</div>'


# ── Tool call trace ────────────────────────────────────────────────────────────

def _tool_trace(record: EvalRecord) -> str:
    if not record.result.tool_calls:
        return '<p class="muted-note">No tool calls made.</p>'
    items = []
    for i, tc in enumerate(record.result.tool_calls, 1):
        args_str = json.dumps(tc.arguments, indent=2)
        result   = tc.result[:500] + ("…" if len(tc.result) > 500 else "")
        items.append(f"""
      <div class="tool-call">
        <div class="tool-header">
          <span class="tool-num">#{i}</span>
          <code class="tool-name">{_h(tc.name)}</code>
        </div>
        <div class="tool-body">
          <div class="tool-col">
            <div class="mini-label">Arguments</div>
            <pre class="code-args">{_h(args_str)}</pre>
          </div>
          <div class="tool-col">
            <div class="mini-label">Result</div>
            <pre class="code-result">{_h(result)}</pre>
          </div>
        </div>
      </div>""")
    return "".join(items)


# ── Per-item accordion ─────────────────────────────────────────────────────────

def _item_row(idx: int, rec: EvalRecord) -> str:
    item   = rec.item
    result = rec.result
    score  = rec.score
    q      = item.input.split("\n")[0][:85]
    n      = len(result.tool_calls)
    chain  = " → ".join(f'<code>{_h(tc.name)}</code>' for tc in result.tool_calls) or "<em class='muted-note'>none</em>"
    rationale = score.details.get("rationale", "")
    err_badge = '<span class="badge red">ERR</span>' if result.error else ""

    return f"""
<details class="item" id="i{idx}">
  <summary class="item-summary">
    <span class="item-num">{idx+1}</span>
    <span class="item-id">{_h(item.id)}</span>
    <span class="item-q">{_h(q)}</span>
    <span class="item-chain">{chain}</span>
    <span class="item-badges">{_badge(score.overall,"overall")} {_badge(score.answer_score,"ans")} {err_badge}</span>
  </summary>
  <div class="item-body">
    <div class="item-grid">
      <div class="item-section">
        <div class="section-label">Question</div>
        <pre class="pre-q">{_h(item.input)}</pre>
      </div>
      <div class="item-section">
        <div class="section-label">Expected answer</div>
        <pre class="pre-expected">{_h(item.expected)}</pre>
      </div>
    </div>
    {"<div class='item-section'><div class='section-label'>Tool call trace (" + str(n) + " call" + ("s" if n!=1 else "") + ")</div>" + _tool_trace(rec) + "</div>" if n > 0 else ""}
    <div class="item-section">
      <div class="section-label">Agent answer</div>
      <pre class="pre-answer">{_h(result.output or "(empty)")}</pre>
    </div>
    <div class="item-section item-scores-row">
      <div class="score-pill">Answer <strong>{score.answer_score:.0%}</strong> <span class="weight">×60%</span></div>
      <div class="score-pill">Reasoning <strong>{score.reasoning_score:.0%}</strong> <span class="weight">×30%</span></div>
      <div class="score-pill">Efficiency <strong>{score.efficiency_score:.0%}</strong> <span class="weight">×10%</span></div>
      <div class="score-pill overall-pill">Overall <strong>{score.overall:.0%}</strong></div>
      {"<div class='rationale'>" + _h(rationale) + "</div>" if rationale else ""}
    </div>
    {"<div class='item-section err-block'><div class='section-label'>Error</div><pre>" + _h(result.error) + "</pre></div>" if result.error else ""}
    <div class="item-meta">
      {"".join(f'<span class="mtag">{_h(k)}: {_h(str(v))}</span>' for k,v in item.metadata.items())}
      <span class="mtag">tokens: {result.total_tokens:,}</span>
      <span class="mtag">{result.latency_ms:.0f} ms</span>
    </div>
  </div>
</details>"""


# ── Single model tab content ───────────────────────────────────────────────────

def _model_tab_body(run: RunResult, tab_id: str) -> str:
    cat   = run.scores_by_category()
    diff  = run.scores_by_difficulty()

    charts = ""
    if cat and len(cat) > 1:
        charts += f'<div class="chart-block"><div class="chart-title">By category</div>{_bar_chart(cat)}</div>'
    if diff and len(diff) > 1:
        charts += f'<div class="chart-block"><div class="chart-title">By difficulty</div>{_bar_chart(diff)}</div>'

    items_html = "\n".join(
        f'<div data-score="{rec.score.overall:.4f}">{_item_row(i, rec)}</div>'
        for i, rec in enumerate(run.records)
    )

    return f"""
<div class="tab-pane" id="{tab_id}">
  <div class="cards">
    {_card("Overall",    run.avg_overall,    "60/30/10 weighted")}
    {_card("Answer",     run.avg_answer,     "correctness")}
    {_card("Reasoning",  run.avg_reasoning,  "chain-of-thought")}
    {_card("Efficiency", run.avg_efficiency, "tool use")}
    {_card_plain("Items",       str(len(run.records)),     "evaluated")}
    {_card_plain("Tokens",      f"{run.total_tokens:,}",   "total")}
    {_card_plain("Avg tools",   f"{run.avg_tool_calls:.1f}", "per item")}
    {_card_err("Error rate",    run.error_rate)}
  </div>

  {"<div class='charts-row'>" + charts + "</div>" if charts else ""}

  <div class="items-header">
    <div class="section-title">Results ({len(run.records)} items)</div>
    <div class="filter-bar">
      <input class="search" type="text" placeholder="Search…" oninput="filterItems(this,'{tab_id}')">
      <button class="fbtn active" onclick="setFilter(this,'{tab_id}','all')">All</button>
      <button class="fbtn" onclick="setFilter(this,'{tab_id}','good')">Good ≥75%</button>
      <button class="fbtn" onclick="setFilter(this,'{tab_id}','mid')">Mid 50-75%</button>
      <button class="fbtn" onclick="setFilter(this,'{tab_id}','low')">Low &lt;50%</button>
    </div>
  </div>
  <div class="items-list" id="list-{tab_id}">
    {items_html}
  </div>
</div>"""


def _card(label: str, score: float, sub: str) -> str:
    c = _score_cls(score)
    return f"""<div class="card card-{c}">
  <div class="card-label">{_h(label)}</div>
  <div class="card-value">{score:.0%}</div>
  <div class="card-sub">{_h(sub)}</div>
</div>"""

def _card_plain(label: str, value: str, sub: str) -> str:
    return f"""<div class="card card-neutral">
  <div class="card-label">{_h(label)}</div>
  <div class="card-value">{_h(value)}</div>
  <div class="card-sub">{_h(sub)}</div>
</div>"""

def _card_err(label: str, rate: float) -> str:
    c = "red" if rate > 0 else "green"
    return f"""<div class="card card-{c}">
  <div class="card-label">{_h(label)}</div>
  <div class="card-value">{rate:.0%}</div>
  <div class="card-sub">failed items</div>
</div>"""


# ── Compare tab ────────────────────────────────────────────────────────────────

MODEL_PALETTE = ["#60a5fa", "#f472b6", "#34d399", "#fb923c", "#a78bfa", "#facc15"]

def _compare_tab(runs: list[RunResult]) -> str:
    colors = {r.agent_name: MODEL_PALETTE[i % len(MODEL_PALETTE)] for i, r in enumerate(runs)}

    # Summary table
    rows = []
    metrics = [
        ("Overall",    lambda r: r.avg_overall),
        ("Answer",     lambda r: r.avg_answer),
        ("Reasoning",  lambda r: r.avg_reasoning),
        ("Efficiency", lambda r: r.avg_efficiency),
        ("Avg tools",  lambda r: r.avg_tool_calls),
        ("Tokens",     lambda r: r.total_tokens),
        ("Errors",     lambda r: r.error_rate),
    ]
    for label, fn in metrics:
        cells = ""
        for run in runs:
            v = fn(run)
            if isinstance(v, float) and label not in ("Avg tools", "Tokens"):
                badge = f'<span class="badge {_score_cls(v)}">{v:.1%}</span>' if label != "Errors" else f'<span class="badge {"red" if v>0 else "green"}">{v:.1%}</span>'
                cells += f"<td>{badge}</td>"
            else:
                cells += f"<td class='num'>{v:,.1f}" + ("</td>" if isinstance(v, float) else f"{v:,}</td>")
        rows.append(f"<tr><td class='metric-name'>{_h(label)}</td>{cells}</tr>")

    headers = "".join(
        f'<th style="color:{colors[r.agent_name]}">{_h(r.agent_name)}</th>'
        for r in runs
    )

    # Per-category comparison
    all_cats: dict[str, dict[str, float]] = {r.agent_name: r.scores_by_category() for r in runs}
    cat_chart = ""
    all_cat_keys: set[str] = set()
    for d in all_cats.values():
        all_cat_keys.update(d.keys())
    if len(all_cat_keys) > 1:
        cat_chart = f"""
    <div class="chart-block">
      <div class="chart-title">Score by category</div>
      {_multi_bar_chart(all_cats, colors)}
    </div>"""

    # Legend
    legend = "".join(
        f'<span class="legend-dot" style="background:{colors[r.agent_name]}"></span><span class="legend-label">{_h(r.agent_name)}</span>'
        for r in runs
    )

    return f"""
<div class="tab-pane" id="tab-compare">
  <div class="legend">{legend}</div>

  <div class="section-title" style="margin-top:0">Side-by-side metrics</div>
  <div class="table-wrap">
    <table class="cmp-table">
      <thead><tr><th>Metric</th>{headers}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>

  {cat_chart}
</div>"""


# ── CSS ────────────────────────────────────────────────────────────────────────

_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:14px}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f172a;color:#cbd5e1;line-height:1.6}

.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 80px}

/* Header */
.rpt-header{padding-bottom:18px;margin-bottom:28px;border-bottom:1px solid #1e293b}
.rpt-header h1{font-size:1.4rem;font-weight:700;color:#f1f5f9}
.rpt-header .sub{font-size:.82rem;color:#64748b;margin-top:3px}
.rpt-header .meta{display:flex;gap:20px;flex-wrap:wrap;margin-top:10px}
.rpt-meta{font-size:.78rem;color:#475569}
.rpt-meta strong{color:#94a3b8}

/* Tabs */
.tabs{display:flex;gap:2px;border-bottom:1px solid #1e293b;margin-bottom:24px;flex-wrap:wrap}
.tab-btn{padding:7px 16px;border-radius:6px 6px 0 0;background:none;border:1px solid transparent;border-bottom:none;color:#64748b;cursor:pointer;font-size:.82rem;font-weight:500;transition:all .15s;margin-bottom:-1px}
.tab-btn:hover{color:#94a3b8;background:#1e293b}
.tab-btn.active{background:#1e293b;border-color:#334155;color:#e2e8f0}
.tab-pane{display:none}.tab-pane.active{display:block}

/* Cards */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-bottom:24px}
.card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px 16px}
.card-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:#64748b;margin-bottom:4px}
.card-value{font-size:1.7rem;font-weight:700;line-height:1}
.card-sub{font-size:.72rem;color:#475569;margin-top:3px}
.card-green .card-value{color:#4ade80}
.card-yellow .card-value{color:#facc15}
.card-red .card-value{color:#f87171}
.card-neutral .card-value{color:#93c5fd}

/* Bar charts */
.charts-row{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:24px}
.chart-block{flex:1;min-width:220px;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:16px}
.chart-title{font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;color:#64748b;margin-bottom:12px}
.bar-chart{display:flex;flex-direction:column;gap:8px}
.bar-row{display:grid;grid-template-columns:110px 1fr 44px;gap:8px;align-items:center}
.bar-label{font-size:.78rem;color:#94a3b8;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar-track{background:#0f172a;border-radius:4px;height:20px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;transition:width .3s ease}
.bar-fill.green{background:#22c55e}
.bar-fill.yellow{background:#eab308}
.bar-fill.red{background:#ef4444}
.bar-pct{font-size:.75rem;color:#64748b;font-weight:600;text-align:right}

/* Multi-bar */
.multi-bar-chart{display:flex;flex-direction:column;gap:16px}
.multi-bar-group{}
.multi-bar-cat{font-size:.78rem;color:#94a3b8;font-weight:600;margin-bottom:6px}
.multi-bar-row{display:grid;grid-template-columns:90px 1fr 44px;gap:6px;align-items:center;margin-bottom:4px}
.multi-bar-model{font-size:.72rem;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* Badges */
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.72rem;font-weight:600;white-space:nowrap}
.badge.green{background:#14532d;color:#4ade80;border:1px solid #166534}
.badge.yellow{background:#422006;color:#facc15;border:1px solid #78350f}
.badge.red{background:#450a0a;color:#f87171;border:1px solid #7f1d1d}

/* Filter / items header */
.items-header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:12px}
.section-title{font-size:.88rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:16px}
.filter-bar{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.search{background:#1e293b;border:1px solid #334155;border-radius:6px;color:#e2e8f0;padding:5px 10px;font-size:.8rem;outline:none;width:180px}
.search:focus{border-color:#3b82f6}
.fbtn{background:#1e293b;border:1px solid #334155;border-radius:6px;color:#64748b;padding:4px 10px;font-size:.76rem;cursor:pointer;transition:all .1s}
.fbtn:hover{background:#334155;color:#e2e8f0}
.fbtn.active{background:#1d4ed8;border-color:#3b82f6;color:#fff}

/* Items list */
.items-list{display:flex;flex-direction:column;gap:4px}
.item{background:#1e293b;border:1px solid #334155;border-radius:7px;overflow:hidden}
.item[open]{border-color:#475569}
.item[open] .item-summary{background:#253347}
.item-summary{display:grid;grid-template-columns:28px 80px 1fr 160px auto;gap:8px;align-items:center;padding:9px 14px;cursor:pointer;list-style:none;user-select:none}
.item-summary::-webkit-details-marker{display:none}
.item-summary:hover{background:#253347}
.item-num{font-size:.72rem;color:#475569;font-weight:600}
.item-id{font-family:monospace;font-size:.76rem;color:#7dd3fc;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.item-q{font-size:.8rem;color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.item-chain{font-size:.72rem;color:#475569;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.item-badges{display:flex;gap:4px;justify-content:flex-end;flex-wrap:nowrap}
.hidden{display:none!important}

/* Item body */
.item-body{padding:16px;border-top:1px solid #1e293b;display:flex;flex-direction:column;gap:14px}
.item-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.item-section{}
.section-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;color:#475569;margin-bottom:6px;font-weight:600}
pre{background:#0f172a;border:1px solid #1e293b;border-radius:5px;padding:10px 12px;font-size:.78rem;overflow-x:auto;white-space:pre-wrap;word-break:break-word;color:#cbd5e1;line-height:1.5;max-height:180px;overflow-y:auto}
.pre-q{max-height:140px}
.pre-expected{color:#86efac}
.pre-answer{max-height:200px}

/* Tool calls */
.tool-call{border:1px solid #1e3a5f;border-radius:6px;overflow:hidden;margin-bottom:8px}
.tool-header{display:flex;align-items:center;gap:8px;padding:6px 12px;background:#0c2444}
.tool-num{font-size:.68rem;background:#1e3a5f;color:#7dd3fc;padding:1px 5px;border-radius:3px;font-weight:700}
.tool-name{font-size:.84rem;color:#38bdf8;font-weight:600}
.tool-body{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:10px 12px;background:#0d1f38}
.tool-col{}
.mini-label{font-size:.65rem;text-transform:uppercase;letter-spacing:.06em;color:#475569;margin-bottom:4px}
.code-args{color:#93c5fd;border-color:#1e3a5f;background:#0a1628;max-height:120px}
.code-result{color:#a7f3d0;border-color:#064e3b;background:#0a1628;max-height:120px}

/* Score pills */
.item-scores-row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.score-pill{background:#0f172a;border:1px solid #334155;border-radius:6px;padding:5px 10px;font-size:.78rem;color:#64748b}
.score-pill strong{color:#e2e8f0}
.overall-pill{border-color:#475569}
.weight{font-size:.68rem;color:#475569}
.rationale{font-size:.78rem;color:#94a3b8;font-style:italic;padding:8px 10px;background:#0f172a;border-left:3px solid #334155;border-radius:0 4px 4px 0;width:100%}

/* Meta / error */
.item-meta{display:flex;flex-wrap:wrap;gap:5px}
.mtag{font-size:.68rem;color:#475569;background:#0f172a;border:1px solid #1e293b;padding:2px 7px;border-radius:4px}
.err-block pre{color:#fca5a5;background:#1c0505;border-color:#7f1d1d}
.muted-note{color:#475569;font-size:.8rem;font-style:italic}

/* Compare table */
.table-wrap{overflow-x:auto;margin-bottom:24px}
.cmp-table{width:100%;border-collapse:collapse;font-size:.82rem}
.cmp-table th{padding:8px 12px;text-align:left;border-bottom:2px solid #334155;font-weight:600;font-size:.76rem;text-transform:uppercase;letter-spacing:.04em}
.cmp-table td{padding:8px 12px;border-bottom:1px solid #1e293b}
.metric-name{color:#94a3b8;font-weight:500}
.num{color:#93c5fd;font-variant-numeric:tabular-nums}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px;align-items:center}
.legend-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}
.legend-label{font-size:.8rem;color:#94a3b8}

@media(max-width:640px){
  .item-summary{grid-template-columns:24px 1fr auto}
  .item-id,.item-chain{display:none}
  .item-grid,.tool-body{grid-template-columns:1fr}
}
"""

# ── JS ─────────────────────────────────────────────────────────────────────────

_JS = """
function showTab(id) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === id));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === id));
}
function filterItems(input, tabId) {
  const q = input.value.toLowerCase();
  const list = document.getElementById('list-' + tabId);
  if (!list) return;
  const filt = list.closest('.tab-pane').querySelector('.fbtn.active')?.dataset?.filt || 'all';
  list.querySelectorAll('[data-score]').forEach(el => {
    const score = parseFloat(el.dataset.score);
    const matchQ = !q || el.textContent.toLowerCase().includes(q);
    const matchF = filt==='all' || (filt==='good' && score>=0.75) || (filt==='mid' && score>=0.5 && score<0.75) || (filt==='low' && score<0.5);
    el.classList.toggle('hidden', !(matchQ && matchF));
  });
}
function setFilter(btn, tabId, filt) {
  const pane = document.getElementById(tabId);
  pane.querySelectorAll('.fbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  btn.dataset.filt = filt;
  const search = pane.querySelector('.search');
  filterItems(search, tabId);
}
// init
document.addEventListener('DOMContentLoaded', () => {
  const first = document.querySelector('.tab-btn');
  if (first) showTab(first.dataset.tab);
});
"""

# ── Public function ────────────────────────────────────────────────────────────

def generate_agent_eval_html(
    runs: RunResult | list[RunResult],
    output_dir: str = "reports",
) -> str:
    """
    Generate a self-contained HTML report for one or more RunResult objects.

    Single model  → flat report (no tabs).
    Multiple models → tabbed: one tab per model + Compare tab.

    Returns absolute path to the saved file.
    """
    if isinstance(runs, RunResult):
        runs = [runs]

    os.makedirs(output_dir, exist_ok=True)

    if len(runs) == 1:
        run      = runs[0]
        filename = f"agent_eval_{run.agent_name}_{run.run_id}.html"
        title    = f"LabAI — {run.agent_name}"
        meta_items = [
            ("Agent", run.agent_name), ("Dataset", run.dataset_name),
            ("Run ID", run.run_id), ("Date", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ]
        tabs_html  = ""
        panes_html = _model_tab_body(run, "tab-main").replace('id="tab-main"', 'id="tab-main" class="tab-pane active"')
        panes_html = panes_html.replace('class="tab-pane" id="tab-main"', 'class="tab-pane active" id="tab-main"')
    else:
        names    = "+".join(r.agent_name for r in runs[:3]) + ("…" if len(runs)>3 else "")
        run_ids  = "_".join(r.run_id for r in runs[:2])
        filename = f"agent_eval_compare_{run_ids}.html"
        title    = f"LabAI — Compare {names}"
        meta_items = [
            ("Models", ", ".join(r.agent_name for r in runs)),
            ("Dataset", runs[0].dataset_name),
            ("Date", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ]

        tab_btns = []
        panes    = []
        for i, run in enumerate(runs):
            tid   = f"tab-{i}"
            active = "active" if i == 0 else ""
            tab_btns.append(f'<button class="tab-btn {active}" data-tab="{tid}" onclick="showTab(\'{tid}\')">{_h(run.agent_name)}</button>')
            body = _model_tab_body(run, tid)
            body = body.replace(f'class="tab-pane" id="{tid}"', f'class="tab-pane {active}" id="{tid}"')
            panes.append(body)

        tab_btns.append('<button class="tab-btn" data-tab="tab-compare" onclick="showTab(\'tab-compare\')">⇄ Compare</button>')
        panes.append(_compare_tab(runs))

        tabs_html  = '<div class="tabs">' + "".join(tab_btns) + "</div>"
        panes_html = "\n".join(panes)

    meta_html = "".join(f'<div class="rpt-meta"><strong>{_h(k)}:</strong> {_h(v)}</div>' for k, v in meta_items)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{_h(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="rpt-header">
    <h1>LabAI Agent Evaluation Report</h1>
    <div class="sub">Universidad Austral | AI Department</div>
    <div class="meta">{meta_html}</div>
  </div>
  {tabs_html}
  {panes_html}
</div>
<script>{_JS}</script>
</body>
</html>"""

    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return os.path.abspath(path)
