#!/usr/bin/env python3
"""
LabAI - Level 2: Agent Evaluation Framework

Evaluates one or more LLM agents against a dataset using an LLM-as-judge scorer.
Results are logged to Braintrust and saved as a PDF report.

Registered components (extend without touching this file):
  Datasets : finance_qa, mmlu
  Agents   : llm_agent
  Scorers  : llm_judge
  Tools    : get_stock_price, get_financial_ratios, calculate_return, compare_companies

Usage:
  python eval_agents.py
  python eval_agents.py --models gpt-4o-mini claude-haiku --samples 10
  python eval_agents.py --dataset finance_qa --categories valuation returns
  python eval_agents.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# ── Import all modules so decorators run and components get registered ─────────
import labai.datasets.finance  # noqa: F401
import labai.datasets.mmlu     # noqa: F401
import labai.tools.finance     # noqa: F401
import labai.agents.llm_agent  # noqa: F401
import labai.scorers.llm_judge # noqa: F401

from labai.core.registry import Registry
from labai.core.runner import AgentEvalRunner
from labai.core.types import RunResult
from labai.agents.llm_agent import LLMAgent
from labai.reports.pdf import generate_agent_eval_pdf
from labai.tools.finance import (
    StockPriceTool,
    FinancialRatiosTool,
    CalculateReturnTool,
    CompareCompaniesTool,
)
from labai.scorers.llm_judge import LLMJudgeScorer

# ── Available models (same as mmlu_benchmark.py) ──────────────────────────────

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

DEFAULT_MODELS   = ["gpt-4o-mini", "claude-haiku"]
DEFAULT_DATASET  = "finance_qa"
DEFAULT_SAMPLES  = 20
DEFAULT_JUDGE    = "openai/gpt-4o-mini"

# Finance tools available to agents
FINANCE_TOOLS = [
    StockPriceTool(),
    FinancialRatiosTool(),
    CalculateReturnTool(),
    CompareCompaniesTool(),
]


# ── Run a single agent ─────────────────────────────────────────────────────────

async def run_eval(
    model_key:   str,
    dataset_key: str,
    categories:  list[str] | None,
    n_samples:   int,
    judge_model: str,
    log_braintrust: bool,
) -> RunResult:
    model_id = MODELS[model_key]

    agent   = LLMAgent(
        name     = model_key,
        model_id = model_id,
        tools    = FINANCE_TOOLS,
    )
    dataset = Registry.get_dataset(dataset_key)()
    scorer  = LLMJudgeScorer(judge_model=judge_model)

    runner = AgentEvalRunner(
        agent              = agent,
        dataset            = dataset,
        scorer             = scorer,
        braintrust_project = "LabAI-Agent-Eval" if log_braintrust else None,
        concurrency        = 3,
        subjects           = categories or None,
        n_samples          = n_samples,
    )

    return await runner.run()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description = "LabAI Level 2 - Agent Evaluation Framework",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = __doc__,
    )
    p.add_argument(
        "--models", "-m",
        nargs  = "+",
        choices = list(MODELS),
        default = DEFAULT_MODELS,
        metavar = "MODEL",
        help    = f"Model keys to evaluate (default: {DEFAULT_MODELS})",
    )
    p.add_argument(
        "--dataset", "-d",
        choices = Registry.list_datasets(),
        default = DEFAULT_DATASET,
        help    = f"Dataset to use (default: {DEFAULT_DATASET})",
    )
    p.add_argument(
        "--categories", "-c",
        nargs   = "+",
        default = None,
        metavar = "CATEGORY",
        help    = "Filter by category keywords (e.g. valuation returns)",
    )
    p.add_argument(
        "--samples", "-n",
        type    = int,
        default = DEFAULT_SAMPLES,
        help    = f"Number of questions per agent (default: {DEFAULT_SAMPLES})",
    )
    p.add_argument(
        "--judge",
        default = DEFAULT_JUDGE,
        help    = f"Judge model (litellm format, default: {DEFAULT_JUDGE})",
    )
    p.add_argument(
        "--no-braintrust",
        action  = "store_true",
        help    = "Disable Braintrust logging",
    )
    p.add_argument(
        "--no-pdf",
        action  = "store_true",
        help    = "Skip PDF report generation",
    )
    p.add_argument(
        "--list",
        action  = "store_true",
        help    = "List registered components and available models",
    )
    return p.parse_args()


def print_registry_summary():
    print("\nRegistered components:")
    summary = Registry.summary()
    for kind, names in summary.items():
        print(f"  {kind:10}: {', '.join(names) or '(none)'}")
    print("\nAvailable models:")
    for k, v in MODELS.items():
        print(f"  {k:<20} {v}")


def print_run_summary(results: list[RunResult]):
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  {'Agent':<22} {'Overall':>8} {'Answer':>8} {'Reasoning':>10} {'Efficiency':>11} {'Tools':>6}")
    print("  " + "-" * 67)
    for r in sorted(results, key=lambda x: x.avg_overall, reverse=True):
        print(
            f"  {r.agent_name:<22} "
            f"{r.avg_overall:>7.1%}  "
            f"{r.avg_answer:>7.1%}  "
            f"{r.avg_reasoning:>9.1%}  "
            f"{r.avg_efficiency:>10.1%}  "
            f"{r.avg_tool_calls:>5.1f}"
        )
    print("=" * 70)


async def main():
    args = parse_args()

    if args.list:
        print_registry_summary()
        return

    if not os.getenv("BRAINTRUST_API_KEY") and not args.no_braintrust:
        print("WARNING: BRAINTRUST_API_KEY not set. Run with --no-braintrust to suppress this.")

    log_bt = not args.no_braintrust and bool(os.getenv("BRAINTRUST_API_KEY"))

    print(f"\nLabAI - Agent Evaluation")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Models    : {', '.join(args.models)}")
    print(f"  Samples   : {args.samples} per agent")
    print(f"  Judge     : {args.judge}")
    print(f"  Braintrust: {'enabled' if log_bt else 'disabled'}")
    if args.categories:
        print(f"  Categories: {', '.join(args.categories)}")

    results: list[RunResult] = []

    for model_key in args.models:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_key}")
        print(f"{'='*60}")
        run = await run_eval(
            model_key      = model_key,
            dataset_key    = args.dataset,
            categories     = args.categories,
            n_samples      = args.samples,
            judge_model    = args.judge,
            log_braintrust = log_bt,
        )
        results.append(run)
        print(
            f"\n  {model_key}: overall={run.avg_overall:.1%}  "
            f"answer={run.avg_answer:.1%}  "
            f"tokens={run.total_tokens:,}"
        )

    print_run_summary(results)

    if not args.no_pdf:
        for run in results:
            path = generate_agent_eval_pdf(run)
            print(f"\n  PDF report: {path}")

    if log_bt:
        print(f"\n  Braintrust: https://www.braintrust.dev/app")


if __name__ == "__main__":
    asyncio.run(main())
