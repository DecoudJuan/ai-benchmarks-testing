"""
PDF report generator for agent benchmark runs.
White background, clean typography — mirrors the HTML report structure.
Filename format: agent_benchmark_<name>_YY-MM-DD_HH-MM.pdf
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
_PURPLE = (139, 92,  246)

def _score_rgb(s: float):
    return _GREEN if s >= 0.75 else (_YELLOW if s >= 0.50 else _RED)

def _fmt_cost(cost: float) -> str:
    if cost <= 0:
        return "$0.00"
    if cost < 0.0001:
        return "< $0.0001"
    return f"${cost:.4f}"

def _latin(t: str) -> str:
    return (str(t)
        .replace("—", "-").replace("–", "-")
        .replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
        .replace("Á", "A").replace("É", "E").replace("Í", "I")
        .replace("Ó", "O").replace("Ú", "U").replace("Ñ", "N")
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
        self.cell(0, 5, "Agent Benchmark  |  Universidad Austral  |  Dept. IA", align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_MID)
        self.cell(0, 6, f"Pagina {self.page_no()}", align="C")

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
        self.cell(52, 6, _latin(label + ":"))
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
        self.set_fill_color(*_BOR)
        self.rect(x0, y0 + 1, bar_w, 4, "F")
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
        self.cell(0, 9, "Agent Benchmark Report", ln=True)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_MID)
        self.cell(0, 5, f"Run {self.run.run_id}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

        self._section("Informacion del run")
        self._kv("Agente",            self.run.agent_name)
        self._kv("Dataset",           self.run.dataset_name)
        self._kv("Items evaluados",   str(len(self.run.records)))
        self._kv("Latencia promedio", f"{self.run.avg_latency_ms:.0f} ms")

        self._section("Scores agregados")
        for label, val in [
            ("Overall (ponderado)",  self.run.avg_overall),
            ("Correctitud respuesta", self.run.avg_answer),
            ("Calidad razonamiento", self.run.avg_reasoning),
            ("Eficiencia tools",     self.run.avg_efficiency),
        ]:
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*_INK)
            self.cell(60, 7, _latin(label))
            self._score_cell(val)
            self.ln()

        self._section("Uso de tokens y costo")
        self._kv("Tokens agente (total)",  f"{self.run.total_tokens:,}")
        self._kv("Avg tool calls/item",    f"{self.run.avg_tool_calls:.2f}")
        self._kv("Tasa de error",          f"{self.run.error_rate:.1%}")
        self.ln(2)

        # Cost breakdown box
        self.set_fill_color(*_BG)
        self.set_draw_color(*_BOR)
        self.set_line_width(0.3)
        box_y = self.get_y()
        self.rect(16, box_y, 178, 26, "FD")
        self.set_y(box_y + 3)

        cost_items = [
            ("Costo agente",  self.run.total_agent_cost),
            ("Costo juez",    self.run.total_judge_cost),
            ("Costo total",   self.run.total_cost),
        ]
        box_w = 59
        for label, cost in cost_items:
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*_MID)
            self.cell(box_w, 5, _latin(label), align="C")
        self.ln()
        for label, cost in cost_items:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*_PURPLE)
            self.cell(box_w, 10, _fmt_cost(cost), align="C")
        self.ln(14)

        # Tools used
        all_tools = sorted({tc.name for rec in self.run.records for tc in rec.result.tool_calls})
        if all_tools:
            self._section("Herramientas del agente")
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*_INK)
            self.cell(0, 6, _latin(", ".join(all_tools)), ln=True)

    def page_breakdown(self):
        cat  = self.run.scores_by_category()
        diff = self.run.scores_by_difficulty()
        if (not cat or len(cat) <= 1) and (not diff or len(diff) <= 1):
            return

        self.add_page()
        self.set_fill_color(*_W)
        self.rect(0, 0, 210, 297, "F")

        if cat and len(cat) > 1:
            self._section("Score por categoria")
            for label, score in sorted(cat.items(), key=lambda x: x[1], reverse=True):
                self._bar(label, score)
            self.ln(2)

        if diff and len(diff) > 1:
            self._section("Score por dificultad")
            for label, score in sorted(diff.items(), key=lambda x: x[1], reverse=True):
                self._bar(label, score)

    def page_items(self):
        self.add_page()
        self.set_fill_color(*_W)
        self.rect(0, 0, 210, 297, "F")

        self._section("Resultados por item")

        # Columns: #, ID, Question, Tools, Overall, Answer, Efficiency, Costo, E
        cols  = [8, 20, 52, 32, 22, 20, 20, 16, 8]
        heads = ["#", "ID", "Pregunta", "Herramientas", "Overall", "Resp.", "Efic.", "Costo", "E"]

        def _thead():
            self.set_fill_color(*_INK)
            self.set_text_color(*_W)
            self.set_font("Helvetica", "B", 7)
            for w, h in zip(cols, heads):
                self.cell(w, 7, h, fill=True, border=1, align="C")
            self.ln()

        _thead()

        for i, rec in enumerate(self.run.records):
            if self.get_y() > 265:
                self.add_page()
                self.set_fill_color(*_W)
                self.rect(0, 0, 210, 297, "F")
                _thead()

            fill  = i % 2 == 0
            score = rec.score
            err   = "Y" if rec.result.error else ""
            q_txt = _latin(rec.item.input.split("\n")[0][:42])

            # Tools used — compact
            tools_txt = _latin(", ".join(tc.name for tc in rec.result.tool_calls) or "-")
            if len(tools_txt) > 24:
                tools_txt = tools_txt[:22] + ".."

            item_cost = rec.result.total_cost + rec.score.details.get("judge_cost", 0.0)

            self._plain_cell(str(i + 1),          cols[0], fill=fill, align="C")
            self._plain_cell(rec.item.id[:16],    cols[1], fill=fill)
            self._plain_cell(q_txt,               cols[2], fill=fill)
            self._plain_cell(tools_txt,           cols[3], fill=fill)
            self._score_cell(score.overall,       cols[4])
            self._score_cell(score.answer_score,  cols[5])
            self._score_cell(score.efficiency_score, cols[6])

            # Cost cell
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*_PURPLE)
            self.set_fill_color(*(_BG if fill else _W))
            self.cell(cols[7], 7, _fmt_cost(item_cost), fill=True, border=1, align="C")
            self.set_text_color(*_INK)

            if err:
                self.set_fill_color(*_RED)
                self.set_text_color(*_W)
                self.set_font("Helvetica", "B", 7)
            else:
                self.set_fill_color(*(_BG if fill else _W))
                self.set_text_color(*_MID)
                self.set_font("Helvetica", "", 7)
            self.cell(cols[8], 7, err, fill=True, border=1, align="C")
            self.set_text_color(*_INK)
            self.ln()

        # Reasoning samples — top 3 lowest scoring items
        self.ln(6)
        self._section("Muestra de razonamiento (3 items con menor score)")
        worst = sorted(self.run.records, key=lambda r: r.score.overall)[:3]
        cell_w = 178  # explicit width = page width - margins (avoids X-position drift)
        for rec in worst:
            if self.get_y() > 240:
                self.add_page()
                self.set_fill_color(*_W)
                self.rect(0, 0, 210, 297, "F")

            self.set_x(self.l_margin)
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*_ACCENT)
            self.cell(cell_w, 6, _latin(f"[{rec.item.id}]  overall={rec.score.overall:.1%}"), ln=True)

            rationale = rec.score.details.get("rationale", "")
            if rationale:
                self.set_x(self.l_margin)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(*_MID)
                self.multi_cell(cell_w, 5, _latin(f"Juez: {rationale[:250]}"))

            answer_preview = (rec.result.output or "")[:300]
            self.set_x(self.l_margin)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*_INK)
            self.multi_cell(cell_w, 5, _latin(f"Agente: {answer_preview}"))
            self.ln(3)


# ── Public ─────────────────────────────────────────────────────────────────────

def generate_agent_eval_pdf(run: RunResult, output_dir: str = "reports") -> str:
    """
    Generate a clean white PDF agent benchmark report.
    Filename: agent_benchmark_<name>_YY-MM-DD_HH-MM.pdf
    Returns absolute path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts       = datetime.now().strftime("%H-%M_%y-%m-%d")
    filename = f"agent_benchmark_{run.agent_name}_{ts}.pdf"
    path     = os.path.join(output_dir, filename)

    pdf = _PDF(run)
    pdf.page_summary()
    pdf.page_breakdown()
    pdf.page_items()
    pdf.output(path)
    return os.path.abspath(path)
