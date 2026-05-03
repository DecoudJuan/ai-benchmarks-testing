#!/usr/bin/env python3
"""
MMLU Benchmark - Braintrust + LLM-as-Judge + PDF Report

Usage:
  python mmlu_benchmark.py                                         # default models, 100 samples
  python mmlu_benchmark.py --models gpt-4o-mini claude-haiku       # specific models
  python mmlu_benchmark.py --subjects math history --samples 200   # filter subjects
  python mmlu_benchmark.py --judge gpt-4o                          # change judge model
  python mmlu_benchmark.py --list-models                           # show available models
  python mmlu_benchmark.py --list-subjects                         # show all MMLU subjects
"""

import argparse
import asyncio
import math
import os
import re
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import braintrust
import litellm
from braintrust import Eval
from datasets import load_dataset

litellm.set_verbose = False
litellm.suppress_debug_info = True

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS: dict[str, str] = {
    "claude-haiku":   "openrouter/anthropic/claude-3.5-haiku",
    "claude-sonnet":  "openrouter/anthropic/claude-sonnet-4-5",
    "claude-opus":    "openrouter/anthropic/claude-opus-4-5",
    "gpt-4o-mini":    "openai/gpt-4o-mini",
    "gpt-4o":         "openai/gpt-4o",
    "llama-3.1-8b":   "openrouter/meta-llama/llama-3.1-8b-instruct",
    "llama-3.1-70b":  "openrouter/meta-llama/llama-3.1-70b-instruct",
    "llama-3.3-70b":  "openrouter/meta-llama/llama-3.3-70b-instruct",
    "gemini-flash":   "openrouter/google/gemini-2.0-flash-001",
    "gemini-pro":     "openrouter/google/gemini-2.5-pro-preview-03-25",
    "mistral-nemo":   "openrouter/mistralai/mistral-nemo",
    "mixtral-8x7b":   "openrouter/mistralai/mixtral-8x7b-instruct",
    "qwen-2.5-72b":   "openrouter/qwen/qwen-2.5-72b-instruct:nitro",
    "deepseek-r1":    "openrouter/deepseek/deepseek-r1",
    "deepseek-v3":    "openrouter/deepseek/deepseek-chat-v3-0324",
    "phi-4":          "openrouter/microsoft/phi-4",
}

DEFAULT_MODELS = ["gpt-4o-mini", "claude-haiku", "gemini-flash", "qwen-2.5-72b"]
DEFAULT_JUDGE  = "openai/gpt-4o-mini"

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history",
    "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    model_key: str
    model_id: str
    judge_scores:    list[float] = field(default_factory=list)
    accuracy_scores: list[float] = field(default_factory=list)
    subjects:        list[str]   = field(default_factory=list)
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0

    @property
    def avg_judge(self) -> float:
        return sum(self.judge_scores) / len(self.judge_scores) if self.judge_scores else 0.0

    @property
    def avg_accuracy(self) -> float:
        return sum(self.accuracy_scores) / len(self.accuracy_scores) if self.accuracy_scores else 0.0

    def by_subject(self) -> dict[str, float]:
        d: dict[str, list[float]] = defaultdict(list)
        for subj, score in zip(self.subjects, self.judge_scores):
            d[subj].append(score)
        return {s: sum(v) / len(v) for s, v in sorted(d.items())}

    def top_subjects(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.by_subject().items(), key=lambda x: x[1], reverse=True)[:n]

    def bottom_subjects(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.by_subject().items(), key=lambda x: x[1])[:n]


# ── Dataset ────────────────────────────────────────────────────────────────────

def format_prompt(example: dict) -> str:
    letters = ["A", "B", "C", "D"]
    choices = "\n".join(f"{letters[i]}. {example['choices'][i]}" for i in range(4))
    return (
        "Answer the following multiple choice question.\n"
        "First explain your reasoning briefly, then end your answer with "
        "ANSWER: X where X is A, B, C or D.\n\n"
        f"Question: {example['question']}\n\n{choices}"
    )


def extract_answer(text: str) -> str:
    m = re.search(r'ANSWER:\s*([ABCD])', text.upper())
    if m:
        return m.group(1)
    matches = re.findall(r'\b([ABCD])\b', text.upper())
    return matches[-1] if matches else ""


def load_mmlu(subjects: list[str] | None, n_samples: int) -> list[dict]:
    print("Loading MMLU dataset from HuggingFace...")
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)

    if subjects:
        matched = [s for s in MMLU_SUBJECTS if any(sub.lower() in s for sub in subjects)]
        if not matched:
            print(f"  Warning: no subjects matched {subjects}. Using all.")
        else:
            ds = ds.filter(lambda x: x["subject"] in matched)
            print(f"  Subjects: {matched}")

    letters = ["A", "B", "C", "D"]
    data = [
        {
            "input":    format_prompt(ex),
            "expected": letters[ex["answer"]],
            "metadata": {"subject": ex["subject"]},
        }
        for ex in ds
    ]

    if n_samples and n_samples < len(data):
        by_subj: dict[str, list] = defaultdict(list)
        for item in data:
            by_subj[item["metadata"]["subject"]].append(item)
        per_subj = max(1, math.ceil(n_samples / len(by_subj)))
        sampled: list[dict] = []
        for items in by_subj.values():
            sampled.extend(items[:per_subj])
        data = sampled[:n_samples]

    print(f"  Loaded {len(data)} questions across {len({d['metadata']['subject'] for d in data})} subjects\n")
    return data


# ── LLM-as-Judge ───────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are evaluating an AI assistant's answer to a multiple choice question.

{question_block}

Correct answer: {expected}

AI's response:
{output}

Score the response from 0.0 to 1.0 using these criteria:
- 1.0  correct answer + clear, accurate reasoning
- 0.7  correct answer + weak or missing reasoning
- 0.4  wrong answer but shows partial understanding of the topic
- 0.0  wrong answer with no relevant reasoning

Reply with ONLY a decimal number between 0.0 and 1.0."""


async def call_judge(question_block: str, output: str, expected: str, judge_model: str) -> float:
    prompt = JUDGE_PROMPT.format(
        question_block=question_block,
        expected=expected,
        output=output or "(no response)",
    )
    try:
        resp = await litellm.acompletion(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "0").strip()
        m = re.search(r'[\d.]+', text)
        return max(0.0, min(1.0, float(m.group()))) if m else 0.0
    except Exception:
        return 0.0


# ── Benchmark runner ────────────────────────────────────────────────────────────

async def run_model(
    model_key: str,
    model_id: str,
    data: list[dict],
    judge_model: str,
    result: ModelResult,
):
    print(f"Evaluating: {model_key}  ({model_id})")
    input_to_subject = {d["input"]: d["metadata"]["subject"] for d in data}

    async def task(input: str) -> str:
        messages = [{"role": "user", "content": input}]
        try:
            with braintrust.current_span().start_span(name="llm_call", type="llm") as span:
                resp = await litellm.acompletion(
                    model=model_id,
                    messages=messages,
                    max_tokens=300,
                    temperature=0,
                )
                raw = resp.choices[0].message.content or ""
                u = resp.usage
                pt = (u.prompt_tokens or 0) if u else 0
                ct = (u.completion_tokens or 0) if u else 0
                tt = (u.total_tokens or pt + ct) if u else 0

                result.prompt_tokens     += pt
                result.completion_tokens += ct
                result.total_tokens      += tt

                span.log(
                    input=messages,
                    output=raw,
                    metadata={"model": model_id},
                    metrics={"prompt_tokens": pt, "completion_tokens": ct, "tokens": tt},
                )
            return raw
        except Exception as e:
            print(f"  [error] {model_key}: {e}")
            return ""

    def accuracy(output: str, expected: str) -> float:
        return 1.0 if extract_answer(output) == expected else 0.0

    async def judge(input: str, output: str, expected: str) -> float:
        score = await call_judge(input, output, expected, judge_model)
        result.judge_scores.append(score)
        result.accuracy_scores.append(accuracy(output, expected))
        result.subjects.append(input_to_subject.get(input, "unknown"))
        return score

    await Eval(
        "MMLU Benchmark",
        data=lambda: data,
        task=task,
        scores=[accuracy, judge],
        metadata={"model": model_key, "model_id": model_id, "judge": judge_model},
    )


# ── Console summary ─────────────────────────────────────────────────────────────

def print_summary(results: list[ModelResult]):
    ranked = sorted(results, key=lambda r: r.avg_judge, reverse=True)

    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  {'Model':<20} {'Judge':>8} {'Accuracy':>10} {'Prompt':>10} {'Completion':>12} {'Total':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    for r in ranked:
        print(
            f"  {r.model_key:<20} {r.avg_judge:>7.1%} {r.avg_accuracy:>10.1%}"
            f" {r.prompt_tokens:>10,} {r.completion_tokens:>12,} {r.total_tokens:>10,}"
        )
    print()


# ── PDF report ──────────────────────────────────────────────────────────────────

BRAND_COLOR  = (41, 82, 163)   # dark blue
ACCENT_COLOR = (230, 236, 255) # light blue
WHITE        = (255, 255, 255)
BLACK        = (30, 30, 30)
GRAY         = (120, 120, 120)
GREEN        = (34, 139, 34)
RED          = (180, 40, 40)


def _score_color(score: float) -> tuple[int, int, int]:
    if score >= 0.75:
        return (34, 139, 34)
    if score >= 0.5:
        return (200, 140, 0)
    return (180, 40, 40)


def generate_pdf(results: list[ModelResult], run_id: str, judge_model: str, n_samples: int) -> str:
    from fpdf import FPDF

    os.makedirs("reports", exist_ok=True)
    path = f"reports/mmlu_{run_id}.pdf"
    ranked = sorted(results, key=lambda r: r.avg_judge, reverse=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(15, 15, 15)

    # ── helpers ────────────────────────────────────────────────────────────────

    def header(title: str):
        pdf.set_fill_color(*BRAND_COLOR)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 9, f"  {title}", ln=True, fill=True)
        pdf.set_text_color(*BLACK)
        pdf.ln(3)

    def section(title: str):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*BRAND_COLOR)
        pdf.cell(0, 7, title, ln=True)
        pdf.set_draw_color(*BRAND_COLOR)
        pdf.set_line_width(0.4)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 180, pdf.get_y())
        pdf.ln(2)
        pdf.set_text_color(*BLACK)

    def kv(label: str, value: str):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*GRAY)
        pdf.cell(45, 6, label)
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(*BLACK)
        pdf.cell(0, 6, value, ln=True)

    def score_badge(score: float, w: int = 22, h: int = 6):
        r, g, b = _score_color(score)
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(w, h, f"{score:.1%}", border=0, fill=True, align="C")
        pdf.set_text_color(*BLACK)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 1 - Title + Executive Summary
    # ──────────────────────────────────────────────────────────────────────────
    pdf.add_page()

    # Title block
    pdf.set_fill_color(*BRAND_COLOR)
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_y(8)
    pdf.cell(0, 10, "MMLU Benchmark Report", align="C", ln=True)
    pdf.set_font("Helvetica", size=9)
    pdf.cell(0, 6, f"Run: {run_id}   |   Date: {now}   |   Judge: {judge_model}   |   Samples per model: {n_samples}", align="C", ln=True)
    pdf.set_text_color(*BLACK)
    pdf.ln(14)

    # Best model callout
    best = ranked[0]
    pdf.set_fill_color(*ACCENT_COLOR)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, f"  Best model: {best.model_key}   Judge score: {best.avg_judge:.1%}   Accuracy: {best.avg_accuracy:.1%}", ln=True, fill=True)
    pdf.ln(6)

    # Leaderboard table
    section("Leaderboard - all models")
    cols  = ["#", "Model", "Judge Score", "Accuracy", "Prompt Tok.", "Completion Tok.", "Total Tok."]
    widths = [8, 44, 26, 24, 28, 34, 28]

    pdf.set_fill_color(*BRAND_COLOR)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 8)
    for col, w in zip(cols, widths):
        pdf.cell(w, 7, col, border=1, fill=True, align="C")
    pdf.ln()

    for i, r in enumerate(ranked):
        pdf.set_fill_color(*ACCENT_COLOR) if i % 2 == 0 else pdf.set_fill_color(*WHITE)
        pdf.set_font("Helvetica", size=8)
        pdf.set_text_color(*BLACK)

        row_vals = [
            str(i + 1),
            r.model_key,
            "",  # judge score → badge
            f"{r.avg_accuracy:.1%}",
            f"{r.prompt_tokens:,}",
            f"{r.completion_tokens:,}",
            f"{r.total_tokens:,}",
        ]
        for j, (val, w) in enumerate(zip(row_vals, widths)):
            if j == 2:  # judge score badge
                x, y = pdf.get_x(), pdf.get_y()
                pdf.set_fill_color(*ACCENT_COLOR) if i % 2 == 0 else pdf.set_fill_color(*WHITE)
                pdf.cell(w, 7, "", border=1, fill=True)
                saved_x, saved_y = pdf.get_x(), pdf.get_y()
                pdf.set_xy(x + (w - 22) / 2, y + 0.5)
                score_badge(r.avg_judge, w=22, h=6)
                pdf.set_xy(saved_x, saved_y)
            else:
                fill = i % 2 == 0
                pdf.set_fill_color(*ACCENT_COLOR) if fill else pdf.set_fill_color(*WHITE)
                pdf.cell(w, 7, val, border=1, fill=fill, align="C" if j != 1 else "L")
        pdf.ln()

    # ──────────────────────────────────────────────────────────────────────────
    # PAGES 2+ - Per-model detail
    # ──────────────────────────────────────────────────────────────────────────
    for rank, r in enumerate(ranked, 1):
        pdf.add_page()

        # Model header band
        pdf.set_fill_color(*BRAND_COLOR)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 11, f"  #{rank}  {r.model_key}", ln=True, fill=True)
        pdf.set_text_color(*BLACK)
        pdf.ln(4)

        # Key metrics row
        metrics = [
            ("Judge Score",  f"{r.avg_judge:.1%}"),
            ("Accuracy",     f"{r.avg_accuracy:.1%}"),
            ("Prompt Tok.",  f"{r.prompt_tokens:,}"),
            ("Completion",   f"{r.completion_tokens:,}"),
            ("Total Tok.",   f"{r.total_tokens:,}"),
            ("Questions",    str(len(r.judge_scores))),
        ]
        box_w = 30
        for label, val in metrics:
            pdf.set_fill_color(*ACCENT_COLOR)
            pdf.set_draw_color(*BRAND_COLOR)
            pdf.set_font("Helvetica", size=7)
            pdf.set_text_color(*GRAY)
            pdf.cell(box_w, 5, label, border="LTR", fill=True, align="C")
        pdf.ln()
        for label, val in metrics:
            pdf.set_fill_color(*ACCENT_COLOR)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*BRAND_COLOR)
            pdf.cell(box_w, 7, val, border="LBR", fill=True, align="C")
        pdf.ln(8)

        # Model ID
        kv("Model ID:", r.model_id)
        pdf.ln(4)

        # Subject breakdown table (two columns)
        by_subj = r.by_subject()
        if by_subj:
            section("Results by Subject")
            subjects_sorted = sorted(by_subj.items(), key=lambda x: x[1], reverse=True)
            half = math.ceil(len(subjects_sorted) / 2)
            left_col  = subjects_sorted[:half]
            right_col = subjects_sorted[half:]

            # Table header x2
            for _ in range(2):
                pdf.set_fill_color(*BRAND_COLOR)
                pdf.set_text_color(*WHITE)
                pdf.set_font("Helvetica", "B", 7)
                pdf.cell(68, 6, "Subject", border=1, fill=True)
                pdf.cell(22, 6, "Judge", border=1, fill=True, align="C")
                pdf.cell(6,  6, "",      border=0)
            pdf.ln()

            for i in range(half):
                fill = i % 2 == 0
                pdf.set_fill_color(*ACCENT_COLOR) if fill else pdf.set_fill_color(*WHITE)
                pdf.set_text_color(*BLACK)
                pdf.set_font("Helvetica", size=7)

                # Left column
                lsubj, lscore = left_col[i]
                pdf.cell(68, 5, lsubj.replace("_", " ").title(), border=1, fill=fill)
                x, y = pdf.get_x(), pdf.get_y()
                pdf.cell(22, 5, "", border=1, fill=fill)
                saved = (pdf.get_x(), pdf.get_y())
                pdf.set_xy(x + 1, y + 0.5)
                score_badge(lscore, w=20, h=4)
                pdf.set_xy(*saved)
                pdf.cell(6, 5, "", border=0)

                # Right column
                if i < len(right_col):
                    rsubj, rscore = right_col[i]
                    pdf.cell(68, 5, rsubj.replace("_", " ").title(), border=1, fill=fill)
                    x, y = pdf.get_x(), pdf.get_y()
                    pdf.cell(22, 5, "", border=1, fill=fill)
                    saved = (pdf.get_x(), pdf.get_y())
                    pdf.set_xy(x + 1, y + 0.5)
                    score_badge(rscore, w=20, h=4)
                    pdf.set_xy(*saved)
                else:
                    pdf.cell(90, 5, "", border=1, fill=fill)
                pdf.ln()

            pdf.ln(5)

            # Top 5 / Bottom 5 callout
            if len(by_subj) >= 5:
                col_w2 = 88
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(*GREEN)
                pdf.cell(col_w2, 6, "  Top 5 subjects", ln=False)
                pdf.set_text_color(*RED)
                pdf.cell(col_w2, 6, "  Bottom 5 subjects", ln=True)
                pdf.set_text_color(*BLACK)

                top    = r.top_subjects(5)
                bottom = r.bottom_subjects(5)
                for (ts, tv), (bs, bv) in zip(top, bottom):
                    pdf.set_font("Helvetica", size=8)
                    pdf.cell(col_w2, 5, f"  {ts.replace('_',' ').title()}: {tv:.1%}", ln=False)
                    pdf.cell(col_w2, 5, f"  {bs.replace('_',' ').title()}: {bv:.1%}", ln=True)

    # ──────────────────────────────────────────────────────────────────────────
    # LAST PAGE - Score distribution overview
    # ──────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    header("Score Distribution - all models at a glance")

    bar_max_w = 130
    for r in ranked:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*BLACK)
        pdf.cell(40, 6, r.model_key, ln=False)

        bar_w = int(r.avg_judge * bar_max_w)
        rc, gc, bc = _score_color(r.avg_judge)
        pdf.set_fill_color(rc, gc, bc)
        pdf.cell(bar_w, 6, "", fill=True, ln=False)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(bar_max_w - bar_w, 6, "", fill=True, ln=False)

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*BLACK)
        pdf.cell(20, 6, f"  {r.avg_judge:.1%}", ln=True)
        pdf.ln(1)

    # Footer on every page
    pdf.set_auto_page_break(auto=False)
    for page in range(1, pdf.page + 1):
        pdf.page = page
        pdf.set_y(-12)
        pdf.set_font("Helvetica", size=7)
        pdf.set_text_color(*GRAY)
        pdf.cell(0, 5, f"MMLU Benchmark  |  Run {run_id}  |  {now}  |  Page {page}", align="C")

    pdf.output(path)
    return path


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MMLU Benchmark with Braintrust + LLM-as-Judge + PDF Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   choices=list(MODELS.keys()), metavar="MODEL")
    p.add_argument("--subjects", nargs="+", default=None, metavar="SUBJECT")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--judge", default=DEFAULT_JUDGE, metavar="MODEL_ID",
                   help=f"LLM judge model (default: {DEFAULT_JUDGE})")
    p.add_argument("--list-models",   action="store_true")
    p.add_argument("--list-subjects", action="store_true")
    return p.parse_args()


async def main():
    args = parse_args()

    if args.list_models:
        print("Available models:\n")
        for key, val in MODELS.items():
            print(f"  {key:<20}  {val}")
        return

    if args.list_subjects:
        for s in MMLU_SUBJECTS:
            print(f"  {s}")
        return

    if not os.getenv("BRAINTRUST_API_KEY"):
        print("ERROR: BRAINTRUST_API_KEY not set. Get it at https://www.braintrust.dev/app/settings?tab=api-keys")
        sys.exit(1)

    run_id = str(uuid.uuid4())[:8]

    print("\n" + "=" * 65)
    print("  MMLU Benchmark - Braintrust + LLM-as-Judge")
    print(f"  Run ID  : {run_id}")
    print(f"  Models  : {', '.join(args.models)}")
    print(f"  Judge   : {args.judge}")
    print(f"  Samples : {args.samples}")
    print(f"  Subjects: {', '.join(args.subjects) if args.subjects else 'all (57)'}")
    print("=" * 65 + "\n")

    data = load_mmlu(args.subjects, args.samples)
    results: list[ModelResult] = []

    for model_key in args.models:
        result = ModelResult(model_key=model_key, model_id=MODELS[model_key])
        await run_model(model_key, MODELS[model_key], data, args.judge, result)
        results.append(result)

    print_summary(results)

    pdf_path = generate_pdf(results, run_id, args.judge, args.samples)
    print(f"PDF report saved: {pdf_path}")
    print(f"Braintrust:       https://www.braintrust.dev/app\n")


if __name__ == "__main__":
    asyncio.run(main())
