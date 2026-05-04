"""
PDF report generator for Level 3 agent evaluation runs.
White background, clean typography — mirrors the HTML report structure.
"""

from __future__ import annotations

import os
from datetime import datetime

from fpdf import FPDF

from labai.core.types import RunResult

# ── Palette ────────────────────────────────────────────────────────────────────
_W   = (255, 255, 255)
_BG  = (248, 250, 252)
_INK = (15,  23,  42)
_MID = (100, 116, 139)
_BOR = (226, 232, 240)

_GREEN  = (34,  197, 94)
_YELLOW = (234, 179, 8)
_RED    = (239, 68,  56)
_ACCENT = (59,  130, 246)

def _score_rgb(s: float):
    return _GREEN if s >= 0.75 else (_YELLOW if s >= 0.50 else _RED)

def _latin(t: str) -> str:
    return (str(t)
        .replace("—", "-").replace("–", "-")
        .replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .encode("latin-1", errors="replace").decode("latin-1"))


class _PDF(FPDF):
    def __init__(self, run: RunResult):
        super().__init__()
        self.run = run
        self.set_margins(16, 16, 16)
        self.set_auto_page_break(auto=True, margin=18)

    def header(self):
        self.set_fill_color(*_ACCENT)
        self.rect(0, 0, 210, 5, "F")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_MID)
        self.set_y(8)
        self.cell(0, 5, "LabAI Agent Evaluation  |  Universidad Austral", align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_MID)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")

    def _section(self, title: str):
        self.ln(4)
        self.set_draw_color(*_ACCENT)
        self.set_line_width(0.4)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*_INK)
        self.cell(0, 7, _latin(title), border="B", ln=True)
        self.set_line_width(0.2)
        self.set_draw_color(*_BOR)
        self.ln(2)

    def _kv(self, label: str, value: str):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_MID)
        self.cell(44, 6, _latin(label + ":"))
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_INK)
        self.cell(0, 6, _latin(str(value)), ln=True)

    def _score_cell(self, score: float, w: float = 28, h: float = 7):
        r, g, b = _score_rgb(score)
        self.set_fill_color(r, g, b)
        self.set_text_color(*_W)
        self.set_font("Helvetica", "B", 9)
        self.cell(w, h, f"{score:.1%}", fill=True, align="C", border=1)
        self.set_text_color(*_INK)

    def _plain_cell(self, val: str, w: float, h: float = 7, align: str = "L", fill: bool = False):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_INK)
        self.set_fill_color(*(_BG if fill else _W))
        self.cell(w, h, _latin(str(val)), align=align, border=1, fill=fill)

    def _bar(self, label: str, score: float, bar_w: float = 90):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_MID)
        self.cell(52, 6, _latin(label), align="R")
        x0, y0 = self.get_x() + 2, self.get_y()
        # track
        self.set_fill_color(*_BOR)
        self.rect(x0, y0 + 1, bar_w, 4, "F")
        # fill
        r, g, b = _score_rgb(score)
        self.set_fill_color(r, g, b)
        self.rect(x0, y0 + 1, bar_w * score, 4, "F")
        self.set_x(x0 + bar_w + 3)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*_INK)
        self.cell(16, 6, f"{score:.1%}", ln=True)

    # ── Pages ──────────────────────────────────────────────────────────────────

    def page_summary(self):
        self.add_page()
        self.set_fill_color(*_W)
        self.rect(0, 0, 210, 297, "F")

        self.ln(2)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*_INK)
        self.cell(0, 9, "Agent Evaluation Report", ln=True)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_MID)
        self.cell(0, 5, f"Run {self.run.run_id}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

        self._section("Run info")
        self._kv("Agent",   self.run.agent_name)
        self._kv("Dataset", self.run.dataset_name)
        self._kv("Items evaluated", str(len(self.run.records)))

        self._section("Aggregate scores")
        for label, val in [
            ("Overall (weighted)",  self.run.avg_overall),
            ("Answer correctness",  self.run.avg_answer),
            ("Reasoning quality",   self.run.avg_reasoning),
            ("Tool efficiency",     self.run.avg_efficiency),
        ]:
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*_INK)
            self.cell(52, 7, _latin(label))
            self._score_cell(val)
            self.ln()

        self._section("Usage")
        self._kv("Total tokens",   f"{self.run.total_tokens:,}")
        self._kv("Avg tool calls", f"{self.run.avg_tool_calls:.2f} per item")
        self._kv("Error rate",     f"{self.run.error_rate:.1%}")

    def page_breakdown(self):
        cat  = self.run.scores_by_category()
        diff = self.run.scores_by_difficulty()
        if (not cat or len(cat) <= 1) and (not diff or len(diff) <= 1):
            return

        self.add_page()
        self.set_fill_color(*_W)
        self.rect(0, 0, 210, 297, "F")

        if cat and len(cat) > 1:
            self._section("Score by category")
            for label, score in sorted(cat.items(), key=lambda x: x[1], reverse=True):
                self._bar(label, score)
            self.ln(2)

        if diff and len(diff) > 1:
            self._section("Score by difficulty")
            for label, score in sorted(diff.items(), key=lambda x: x[1], reverse=True):
                self._bar(label, score)

    def page_items(self):
        self.add_page()
        self.set_fill_color(*_W)
        self.rect(0, 0, 210, 297, "F")

        self._section("Per-item results")

        cols  = [10, 22, 72, 26, 24, 26, 10]
        heads = ["#", "ID", "Question", "Overall", "Answer", "Efficiency", "E"]

        def _thead():
            self.set_fill_color(*_INK)
            self.set_text_color(*_W)
            self.set_font("Helvetica", "B", 8)
            for w, h in zip(cols, heads):
                self.cell(w, 7, h, fill=True, border=1, align="C")
            self.ln()

        _thead()

        for i, rec in enumerate(self.run.records):
            if self.get_y() > 268:
                self.add_page()
                self.set_fill_color(*_W)
                self.rect(0, 0, 210, 297, "F")
                _thead()

            fill  = i % 2 == 0
            score = rec.score
            err   = "Y" if rec.result.error else ""
            q_txt = _latin(rec.item.input.split("\n")[0][:55])

            self._plain_cell(str(i + 1),                    cols[0], fill=fill, align="C")
            self._plain_cell(rec.item.id[:18],              cols[1], fill=fill)
            self._plain_cell(q_txt,                         cols[2], fill=fill)
            self._score_cell(score.overall,                 cols[3])
            self._score_cell(score.answer_score,            cols[4])
            self._score_cell(score.efficiency_score,        cols[5])

            if err:
                self.set_fill_color(*_RED)
                self.set_text_color(*_W)
                self.set_font("Helvetica", "B", 8)
            else:
                self.set_fill_color(*(_BG if fill else _W))
                self.set_text_color(*_MID)
                self.set_font("Helvetica", "", 8)
            self.cell(cols[6], 7, err, fill=True, border=1, align="C")
            self.set_text_color(*_INK)
            self.ln()


# ── Public ─────────────────────────────────────────────────────────────────────

def generate_agent_eval_pdf(run: RunResult, output_dir: str = "reports") -> str:
    """Generate a clean white PDF report for a single RunResult."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"agent_eval_{run.agent_name}_{run.run_id}.pdf"
    path     = os.path.join(output_dir, filename)

    pdf = _PDF(run)
    pdf.page_summary()
    pdf.page_breakdown()
    pdf.page_items()
    pdf.output(path)
    return os.path.abspath(path)
