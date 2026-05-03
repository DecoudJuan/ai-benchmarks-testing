#!/usr/bin/env python3
"""
MMLU Benchmark with Braintrust

Usage:
  python mmlu_benchmark.py                                         # default models, 100 samples
  python mmlu_benchmark.py --models gpt-4o-mini claude-haiku       # specific models
  python mmlu_benchmark.py --subjects math history --samples 200   # filter subjects
  python mmlu_benchmark.py --list-models                           # show available models
  python mmlu_benchmark.py --list-subjects                         # show all MMLU subjects
"""

import argparse
import asyncio
import os
import re
import sys
from collections import defaultdict

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
    # Anthropic via OpenRouter
    "claude-haiku":       "openrouter/anthropic/claude-3.5-haiku",
    "claude-sonnet":      "openrouter/anthropic/claude-sonnet-4-5",
    "claude-opus":        "openrouter/anthropic/claude-opus-4-5",
    # OpenAI (direct)
    "gpt-4o-mini":        "openai/gpt-4o-mini",
    "gpt-4o":             "openai/gpt-4o",
    # OpenRouter — Meta Llama
    "llama-3.1-8b":       "openrouter/meta-llama/llama-3.1-8b-instruct",
    "llama-3.1-70b":      "openrouter/meta-llama/llama-3.1-70b-instruct",
    "llama-3.3-70b":      "openrouter/meta-llama/llama-3.3-70b-instruct",
    # OpenRouter — Google
    "gemini-flash":       "openrouter/google/gemini-2.0-flash-001",
    "gemini-pro":         "openrouter/google/gemini-2.5-pro-preview-03-25",
    # OpenRouter — Mistral
    "mistral-nemo":       "openrouter/mistralai/mistral-nemo",
    "mixtral-8x7b":       "openrouter/mistralai/mixtral-8x7b-instruct",
    # OpenRouter — others
    "qwen-2.5-72b":       "openrouter/qwen/qwen-2.5-72b-instruct:nitro",
    "deepseek-r1":        "openrouter/deepseek/deepseek-r1",
    "deepseek-v3":        "openrouter/deepseek/deepseek-chat-v3-0324",
    "phi-4":              "openrouter/microsoft/phi-4",
}

DEFAULT_MODELS = ["gpt-4o-mini", "claude-haiku", "gemini-flash", "qwen-2.5-72b"]

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


# ── Dataset helpers ─────────────────────────────────────────────────────────────

def format_prompt(example: dict) -> str:
    letters = ["A", "B", "C", "D"]
    choices = "\n".join(f"{letters[i]}. {example['choices'][i]}" for i in range(4))
    return (
        "Answer the following multiple choice question. "
        "Reply with ONLY the letter of the correct answer (A, B, C, or D).\n\n"
        f"Question: {example['question']}\n\n{choices}"
    )


def extract_answer(text: str) -> str:
    text = text.strip().upper()
    if text and text[0] in "ABCD":
        return text[0]
    m = re.search(r'\b([ABCD])\b', text)
    return m.group(1) if m else ""


def load_mmlu(subjects: list[str] | None, n_samples: int) -> list[dict]:
    print("Loading MMLU dataset from HuggingFace...")
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)

    if subjects:
        matched = [s for s in MMLU_SUBJECTS if any(sub.lower() in s for sub in subjects)]
        if not matched:
            print(f"  Warning: no subjects matched {subjects}. Using all subjects.")
        else:
            ds = ds.filter(lambda x: x["subject"] in matched)
            print(f"  Subjects: {matched}")

    letters = ["A", "B", "C", "D"]
    data = [
        {
            "input": format_prompt(ex),
            "expected": letters[ex["answer"]],
            "metadata": {"subject": ex["subject"]},
        }
        for ex in ds
    ]

    # Sample evenly across subjects so every subject is represented
    if n_samples and n_samples < len(data):
        from collections import defaultdict
        import math
        by_subject: dict[str, list] = defaultdict(list)
        for item in data:
            by_subject[item["metadata"]["subject"]].append(item)

        n_subjects = len(by_subject)
        per_subject = max(1, math.ceil(n_samples / n_subjects))
        sampled = []
        for items in by_subject.values():
            sampled.extend(items[:per_subject])
        data = sampled[:n_samples]

    print(f"  Loaded {len(data)} questions across {len({d['metadata']['subject'] for d in data})} subjects\n")
    return data


# ── Similarity ──────────────────────────────────────────────────────────────────

def print_similarity(all_answers: dict[str, list[str]]):
    """Print pairwise agreement matrix between models."""
    models = list(all_answers.keys())
    if len(models) < 2:
        return

    print("\n" + "=" * 60)
    print("  MODEL SIMILARITY  (% of questions with same answer)")
    print("=" * 60)

    # Header
    col_w = 14
    header = " " * 16 + "".join(f"{m[:col_w]:>{col_w}}" for m in models)
    print(header)

    for m1 in models:
        row = f"  {m1[:14]:<14}"
        answers1 = all_answers[m1]
        for m2 in models:
            answers2 = all_answers[m2]
            n = min(len(answers1), len(answers2))
            if n == 0:
                row += f"{'N/A':>{col_w}}"
            elif m1 == m2:
                row += f"{'100.0%':>{col_w}}"
            else:
                agree = sum(a == b for a, b in zip(answers1, answers2) if a and b)
                total = sum(1 for a, b in zip(answers1, answers2) if a and b)
                pct = (agree / total * 100) if total else 0
                row += f"{f'{pct:.1f}%':>{col_w}}"
        print(row)
    print()


# ── Token summary ───────────────────────────────────────────────────────────────

def print_token_summary(token_totals: dict[str, dict]):
    print("=" * 60)
    print("  TOKEN USAGE")
    print("=" * 60)
    print(f"  {'Model':<20} {'Prompt':>10} {'Completion':>12} {'Total':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10}")
    for model_key, t in token_totals.items():
        print(f"  {model_key:<20} {t['prompt']:>10,} {t['completion']:>12,} {t['total']:>10,}")
    print()


# ── Benchmark ───────────────────────────────────────────────────────────────────

async def run_model(
    model_key: str,
    model_id: str,
    data: list[dict],
    all_answers: dict[str, list[str]],
    token_totals: dict[str, dict],
):
    print(f"Evaluating: {model_key}  ({model_id})")
    answers: list[str] = []
    tokens = {"prompt": 0, "completion": 0, "total": 0}

    async def task(input: str) -> str:
        messages = [{"role": "user", "content": input}]
        try:
            with braintrust.current_span().start_span(name="llm_call", type="llm") as llm_span:
                resp = await litellm.acompletion(
                    model=model_id,
                    messages=messages,
                    max_tokens=16,
                    temperature=0,
                )
                raw = resp.choices[0].message.content or ""
                answer = extract_answer(raw)

                u = resp.usage
                pt = (u.prompt_tokens or 0) if u else 0
                ct = (u.completion_tokens or 0) if u else 0
                tt = (u.total_tokens or pt + ct) if u else 0

                tokens["prompt"] += pt
                tokens["completion"] += ct
                tokens["total"] += tt

                llm_span.log(
                    input=messages,
                    output=raw,
                    metadata={"model": model_id},
                    metrics={
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "tokens": tt,
                    },
                )

            answers.append(answer)
            return answer
        except Exception as e:
            print(f"  [error] {model_key}: {e}")
            answers.append("")
            return ""

    def accuracy(output: str, expected: str) -> float:
        return 1.0 if output and output == expected else 0.0

    await Eval(
        "MMLU Benchmark",
        data=lambda: data,
        task=task,
        scores=[accuracy],
        metadata={"model": model_key, "model_id": model_id},
    )

    all_answers[model_key] = answers
    token_totals[model_key] = tokens


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MMLU Benchmark with Braintrust",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   choices=list(MODELS.keys()), metavar="MODEL",
                   help=f"Models to evaluate (default: {', '.join(DEFAULT_MODELS)})")
    p.add_argument("--subjects", nargs="+", default=None, metavar="SUBJECT",
                   help="Filter MMLU subjects (partial match, e.g. math history)")
    p.add_argument("--samples", type=int, default=100,
                   help="Number of questions to evaluate per model (default: 100)")
    p.add_argument("--list-models", action="store_true",
                   help="Print available models and exit")
    p.add_argument("--list-subjects", action="store_true",
                   help="Print all MMLU subjects and exit")
    return p.parse_args()


async def main():
    args = parse_args()

    if args.list_models:
        print("Available models:\n")
        for key, val in MODELS.items():
            print(f"  {key:<20}  {val}")
        return

    if args.list_subjects:
        print("MMLU subjects:\n")
        for s in MMLU_SUBJECTS:
            print(f"  {s}")
        return

    if not os.getenv("BRAINTRUST_API_KEY"):
        print("ERROR: BRAINTRUST_API_KEY is not set.")
        print("Get your key at https://www.braintrust.dev/app/settings?tab=api-keys")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  MMLU Benchmark — Braintrust")
    print(f"  Models  : {', '.join(args.models)}")
    print(f"  Samples : {args.samples}")
    print(f"  Subjects: {', '.join(args.subjects) if args.subjects else 'all'}")
    print("=" * 60 + "\n")

    data = load_mmlu(args.subjects, args.samples)

    all_answers: dict[str, list[str]] = {}
    token_totals: dict[str, dict] = {}

    for model_key in args.models:
        await run_model(model_key, MODELS[model_key], data, all_answers, token_totals)

    print_token_summary(token_totals)
    print_similarity(all_answers)
    print("Results at https://www.braintrust.dev/app")


if __name__ == "__main__":
    asyncio.run(main())
