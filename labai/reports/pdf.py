"""
PDF report generator for Level 2 agent evaluation runs.

Produces a multi-page PDF with:
  Page 1  — Run summary (agent, dataset, aggregate scores, token usage)
  Page 2+ — Per-category breakdown (avg scores, question count)
  Last    — Per-item detail table (id, overall score, tool calls, error flag)

Uses fpdf2. All text must be Latin-1 safe (no Unicode em-dashes, etc.).
"""

from __future__ import annotations

import os
from datetime import datetime

from fpdf import FPDF

from labai.core.types import RunResult

# ── Colour palette ─────────────────────────────────────────────────────────────
_GREEN  = (34,  197, 94)
_YELLOW = (234, 179, 8)
_RED    = (239, 68,  56)
_DARK   = (30,  41,  59)
_LIGHT  = (248, 250, 252)
_WHITE  = (255, 255, 255)
_BORDER = (203, 213, 225)


def _score_colour(score: float) -> tuple[int, int, int]:
    if score >= 0.75:
        return _GREEN
    if score >= 0.50:
        return _YELLOW
    return _RED


def _latin(text: str) -> str:
    """Replace characters outside Latin-1 with ASCII equivalents."""
    return (
        text
        .replace("—", "-")   # em dash
        .replace("–", "-")   # en dash
        .replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


# ── PDF class ──────────────────────────────────────────────────────────────────

class _AgentReportPDF(FPDF):
    def __init__(self, run: RunResult):
        super().__init__()
        self.run       = run
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(auto=True, margin=20)

    # ── Header / footer ────────────────────────────────────────────────────────
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*_DARK)
        self.cell(0, 8, "LabAI - Agent Evaluation Report | Universidad Austral", align="C")
        self.ln(4)
        self.set_draw_color(*_BORDER)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _section_title(self, text: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*_DARK)
        self.set_fill_color(*_LIGHT)
        self.cell(0, 8, _latin(text), fill=True, ln=True)
        self.ln(2)

    def _kv(self, label: str, value: str, bold_value: bool = False):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*_DARK)
        self.cell(55, 6, _latin(label + ":"), ln=False)
        self.set_font("Helvetica", "B" if bold_value else "", 10)
        self.cell(0, 6, _latin(str(value)), ln=True)

    def _score_cell(self, score: float, w: float = 30):
        r, g, b = _score_colour(score)
        self.set_fill_color(r, g, b)
        self.set_text_color(*_WHITE)
        self.set_font("Helvetica", "B", 10)
        self.cell(w, 7, f"{score:.1%}", fill=True, align="C")
        self.set_text_color(*_DARK)

    # ── Pages ──────────────────────────────────────────────────────────────────
    def page_summary(self):
        """Page 1: run-level summary."""
        self.add_page()
        self._section_title("Run Summary")

        run = self.run
        self._kv("Run ID",       run.run_id)
        self._kv("Agent",        run.agent_name)
        self._kv("Dataset",      run.dataset_name)
        self._kv("Items scored", str(len(run.records)))
        self._kv("Date",         datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.ln(4)

        self._section_title("Aggregate Scores")
        rows = [
            ("Overall",     run.avg_overall),
            ("Answer",      run.avg_answer),
            ("Reasoning",   run.avg_reasoning),
            ("Efficiency",  run.avg_efficiency),
        ]
        for label, val in rows:
            self.set_font("Helvetica", "", 10)
            self.set_text_color(*_DARK)
            self.cell(60, 7, label)
            self._score_cell(val)
            self.ln()

        self.ln(4)
        self._section_title("Token Usage")
        self._kv("Total tokens",      f"{run.total_tokens:,}")
        self._kv("Avg tool calls",    f"{run.avg_tool_calls:.2f}")
        self._kv("Error rate",        f"{run.error_rate:.1%}")

    def page_by_category(self):
        """Page 2: breakdown by category."""
        self.add_page()
        self._section_title("Scores by Category")

        cat_scores = self.run.scores_by_category()
        if not cat_scores:
            self.cell(0, 8, "No category data available.", ln=True)
            return

        # Table header
        self.set_fill_color(*_DARK)
        self.set_text_color(*_WHITE)
        self.set_font("Helvetica", "B", 10)
        self.cell(80, 8, "Category", fill=True, border=1)
        self.cell(40, 8, "Avg Overall", fill=True, border=1, align="C")
        self.cell(40, 8, "Questions",   fill=True, border=1, align="C")
        self.ln()

        # Count per category
        counts: dict[str, int] = {}
        for rec in self.run.records:
            cat = rec.item.metadata.get("category", "unknown")
            counts[cat] = counts.get(cat, 0) + 1

        self.set_text_color(*_DARK)
        for i, (cat, score) in enumerate(sorted(cat_scores.items())):
            fill = i % 2 == 0
            self.set_fill_color(*(248, 250, 252) if fill else (255, 255, 255))
            self.set_font("Helvetica", "", 10)
            self.cell(80, 7, _latin(cat), fill=fill, border=1)
            self._score_cell(score, w=40)
            self.set_fill_color(*(248, 250, 252) if fill else (255, 255, 255))
            self.set_font("Helvetica", "", 10)
            self.cell(40, 7, str(counts.get(cat, 0)), fill=fill, border=1, align="C")
            self.ln()

        self.ln(6)
        self._section_title("Scores by Difficulty")

        diff_scores = self.run.scores_by_difficulty()
        if not diff_scores:
            self.cell(0, 8, "No difficulty data available.", ln=True)
            return

        self.set_fill_color(*_DARK)
        self.set_text_color(*_WHITE)
        self.set_font("Helvetica", "B", 10)
        self.cell(80, 8, "Difficulty", fill=True, border=1)
        self.cell(40, 8, "Avg Overall", fill=True, border=1, align="C")
        self.cell(40, 8, "Questions",   fill=True, border=1, align="C")
        self.ln()

        diff_counts: dict[str, int] = {}
        for rec in self.run.records:
            diff = rec.item.metadata.get("difficulty", "unknown")
            diff_counts[diff] = diff_counts.get(diff, 0) + 1

        self.set_text_color(*_DARK)
        for i, (diff, score) in enumerate(sorted(diff_scores.items())):
            fill = i % 2 == 0
            self.set_fill_color(*(248, 250, 252) if fill else (255, 255, 255))
            self.set_font("Helvetica", "", 10)
            self.cell(80, 7, _latin(diff), fill=fill, border=1)
            self._score_cell(score, w=40)
            self.set_fill_color(*(248, 250, 252) if fill else (255, 255, 255))
            self.set_font("Helvetica", "", 10)
            self.cell(40, 7, str(diff_counts.get(diff, 0)), fill=fill, border=1, align="C")
            self.ln()

    def page_item_detail(self):
        """Page(s) 3+: per-item detail table."""
        self.add_page()
        self._section_title("Per-Item Results")

        # Header
        col_w = [28, 68, 25, 25, 25, 12]
        headers = ["ID", "Input (truncated)", "Overall", "Answer", "Tools", "Err"]

        self.set_fill_color(*_DARK)
        self.set_text_color(*_WHITE)
        self.set_font("Helvetica", "B", 9)
        for w, h in zip(col_w, headers):
            self.cell(w, 7, h, fill=True, border=1, align="C")
        self.ln()

        self.set_text_color(*_DARK)
        for i, rec in enumerate(self.run.records):
            if self.get_y() > 265:
                self.add_page()
                # Re-print header
                self.set_fill_color(*_DARK)
                self.set_text_color(*_WHITE)
                self.set_font("Helvetica", "B", 9)
                for w, h in zip(col_w, headers):
                    self.cell(w, 7, h, fill=True, border=1, align="C")
                self.ln()
                self.set_text_color(*_DARK)

            fill = i % 2 == 0
            bg   = (248, 250, 252) if fill else (255, 255, 255)
            self.set_fill_color(*bg)
            self.set_font("Helvetica", "", 8)

            item_id   = _latin(rec.item.id[:26])
            input_txt = _latin(rec.item.input[:65].replace("\n", " ") + "...")
            n_tools   = str(len(rec.result.tool_calls))
            err_flag  = "Y" if rec.result.error else "N"

            self.cell(col_w[0], 6, item_id,   fill=fill, border=1)
            self.cell(col_w[1], 6, input_txt, fill=fill, border=1)

            # Colour-coded score cells (inline)
            for score, w in [
                (rec.score.overall,      col_w[2]),
                (rec.score.answer_score, col_w[3]),
            ]:
                r, g, b = _score_colour(score)
                self.set_fill_color(r, g, b)
                self.set_text_color(*_WHITE)
                self.set_font("Helvetica", "B", 8)
                self.cell(w, 6, f"{score:.2f}", fill=True, border=1, align="C")
                self.set_text_color(*_DARK)
                self.set_fill_color(*bg)
                self.set_font("Helvetica", "", 8)

            self.cell(col_w[4], 6, n_tools,  fill=fill, border=1, align="C")

            # Error flag
            if err_flag == "Y":
                self.set_fill_color(*_RED)
                self.set_text_color(*_WHITE)
            self.cell(col_w[5], 6, err_flag, fill=True, border=1, align="C")
            self.set_text_color(*_DARK)
            self.set_fill_color(*bg)
            self.ln()

    def page_bar_chart(self):
        """Final page: horizontal bar chart of per-category scores."""
        self.add_page()
        self._section_title("Score Distribution by Category")

        cat_scores = self.run.scores_by_category()
        if not cat_scores:
            return

        sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        bar_area_w  = 120
        bar_h       = 10
        x_start     = 70
        y            = self.get_y() + 4

        for cat, score in sorted_cats:
            bar_w = bar_area_w * score
            r, g, b = _score_colour(score)

            self.set_xy(15, y)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*_DARK)
            self.cell(50, bar_h, _latin(cat), align="R")

            self.set_xy(x_start, y)
            self.set_fill_color(r, g, b)
            self.cell(bar_w, bar_h, "", fill=True)

            self.set_xy(x_start + bar_w + 2, y)
            self.set_font("Helvetica", "B", 9)
            self.cell(20, bar_h, f"{score:.1%}")

            y += bar_h + 3

        # x-axis labels
        self.set_xy(x_start, y + 2)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(120, 120, 120)
        for pct in [0, 25, 50, 75, 100]:
            x = x_start + bar_area_w * pct / 100
            self.set_xy(x - 5, y + 2)
            self.cell(10, 5, f"{pct}%", align="C")


# ── Public function ────────────────────────────────────────────────────────────

def generate_agent_eval_pdf(run: RunResult, output_dir: str = "reports") -> str:
    """
    Generate a PDF report for an agent evaluation RunResult.

    Args:
        run:        The RunResult to report on.
        output_dir: Directory to save the PDF (created if needed).

    Returns:
        Absolute path to the saved PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"agent_eval_{run.agent_name}_{run.run_id}.pdf"
    path     = os.path.join(output_dir, filename)

    pdf = _AgentReportPDF(run)
    pdf.page_summary()
    pdf.page_by_category()
    pdf.page_item_detail()
    pdf.page_bar_chart()
    pdf.output(path)

    return os.path.abspath(path)
