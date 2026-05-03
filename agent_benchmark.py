#!/usr/bin/env python3
"""
LabAI - Level 2: Orchestrator Agent

An AI agent (Claude via OpenRouter) that receives a natural language instruction
and autonomously decides which models to benchmark, on which subjects,
with how many samples, and how to interpret the results.

The agent can:
  - Run benchmarks on specific models / subjects / sample sizes
  - Check intermediate results and decide to dig deeper
  - Compare models after collecting data
  - Generate the PDF report when done

Usage:
  python agent_benchmark.py
  python agent_benchmark.py --instruction "Compare gpt-4o-mini and deepseek-v3 on math, 50 questions each"
  python agent_benchmark.py --orchestrator claude-opus
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import litellm
litellm.set_verbose = False
litellm.suppress_debug_info = True

from mmlu_benchmark import (
    MODELS,
    MMLU_SUBJECTS,
    DEFAULT_JUDGE,
    ModelResult,
    load_mmlu,
    run_model,
    generate_pdf,
    print_summary,
)

# ── Orchestrator models ────────────────────────────────────────────────────────

ORCHESTRATORS = {
    "claude-haiku":  "openrouter/anthropic/claude-3.5-haiku",
    "claude-sonnet": "openrouter/anthropic/claude-sonnet-4-5",
    "claude-opus":   "openrouter/anthropic/claude-opus-4-5",
    "gpt-4o":        "openai/gpt-4o",
    "gpt-4o-mini":   "openai/gpt-4o-mini",
}
DEFAULT_ORCHESTRATOR = "claude-sonnet"

DEFAULT_INSTRUCTION = """
You are a research assistant at LabAI, Universidad Austral.
Run an MMLU benchmark on these models: gpt-4o-mini, claude-haiku, gemini-flash, deepseek-v3.
Use 50 questions per model sampled across all subjects.
After collecting all results, identify:
  1. Which model performs best overall
  2. Which subjects are hardest across all models
  3. Any surprising findings
Then generate the PDF report.
"""

# ── Tool definitions (OpenAI format for litellm) ───────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_available_models",
            "description": "List all LLM models available for benchmarking.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_subjects",
            "description": "List all available MMLU subjects for filtering benchmarks.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_benchmark",
            "description": (
                "Run an MMLU benchmark on a single model with LLM-as-judge scoring. "
                "Returns judge score (0-1), accuracy, and token usage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model key to evaluate (e.g. 'gpt-4o-mini', 'claude-haiku').",
                    },
                    "subjects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional MMLU subject keywords to filter (e.g. ['math', 'physics']). Omit for all subjects.",
                    },
                    "samples": {
                        "type": "integer",
                        "description": "Number of questions (default: 50). Sampled evenly across subjects.",
                    },
                },
                "required": ["model"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_results",
            "description": "Get current benchmark results for all models evaluated so far, ranked by judge score.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_subject_breakdown",
            "description": "Get per-subject judge scores (top 5 and bottom 5) for a specific model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model key to inspect.",
                    },
                },
                "required": ["model"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate a multilevel PDF report with all collected results. Call this when analysis is complete.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ── Tool handlers ──────────────────────────────────────────────────────────────

async def handle_tool(name: str, inputs: dict, session: dict) -> str:
    results: dict[str, ModelResult] = session["results"]

    if name == "list_available_models":
        lines = [f"  {k:<20} {v}" for k, v in MODELS.items()]
        return "Available models:\n" + "\n".join(lines)

    if name == "list_available_subjects":
        return "Available MMLU subjects:\n" + "\n".join(f"  {s}" for s in MMLU_SUBJECTS)

    if name == "run_benchmark":
        model_key = inputs.get("model", "")
        if model_key not in MODELS:
            return f"Error: '{model_key}' is not a valid model key. Use list_available_models."

        subjects = inputs.get("subjects") or None
        samples  = inputs.get("samples", 50)

        print(f"\n  [tool] run_benchmark: {model_key} | subjects={subjects} | samples={samples}")

        data   = load_mmlu(subjects, samples)
        result = ModelResult(model_key=model_key, model_id=MODELS[model_key])
        await run_model(model_key, MODELS[model_key], data, DEFAULT_JUDGE, result)
        results[model_key] = result

        return json.dumps({
            "model":            model_key,
            "judge_score":      f"{result.avg_judge:.1%}",
            "accuracy":         f"{result.avg_accuracy:.1%}",
            "questions_scored": len(result.judge_scores),
            "prompt_tokens":    result.prompt_tokens,
            "completion_tokens":result.completion_tokens,
            "total_tokens":     result.total_tokens,
        }, indent=2)

    if name == "get_results":
        if not results:
            return "No results yet. Use run_benchmark first."
        ranked = sorted(
            results.items(),
            key=lambda x: x[1].avg_judge,
            reverse=True,
        )
        summary = {
            k: {
                "judge_score": f"{r.avg_judge:.1%}",
                "accuracy":    f"{r.avg_accuracy:.1%}",
                "total_tokens": r.total_tokens,
                "questions":   len(r.judge_scores),
            }
            for k, r in ranked
        }
        return json.dumps({"ranked_by_judge_score": summary}, indent=2)

    if name == "get_subject_breakdown":
        model_key = inputs.get("model", "")
        if model_key not in results:
            return f"No results for '{model_key}'. Run benchmark first."
        r = results[model_key]
        by_subj = r.by_subject()
        top    = sorted(by_subj.items(), key=lambda x: x[1], reverse=True)[:5]
        bottom = sorted(by_subj.items(), key=lambda x: x[1])[:5]
        return json.dumps({
            "model":             model_key,
            "top_5_subjects":    {s: f"{v:.1%}" for s, v in top},
            "bottom_5_subjects": {s: f"{v:.1%}" for s, v in bottom},
            "all_subjects":      {s: f"{v:.1%}" for s, v in sorted(by_subj.items())},
        }, indent=2)

    if name == "generate_report":
        if not results:
            return "No results to report. Run at least one benchmark first."
        max_q = max((len(r.judge_scores) for r in results.values()), default=0)
        path  = generate_pdf(list(results.values()), session["run_id"], DEFAULT_JUDGE, max_q)
        return f"PDF report saved to: {path}"

    return f"Unknown tool: {name}"


# ── Agent loop ─────────────────────────────────────────────────────────────────

async def run_agent(instruction: str, orchestrator_key: str):
    model_id = ORCHESTRATORS.get(orchestrator_key, ORCHESTRATORS[DEFAULT_ORCHESTRATOR])
    run_id   = str(uuid.uuid4())[:8]
    session  = {"results": {}, "run_id": run_id}

    print("\n" + "=" * 65)
    print("  LabAI - Orchestrator Agent (Level 2)")
    print(f"  Run ID      : {run_id}")
    print(f"  Orchestrator: {orchestrator_key} ({model_id})")
    print(f"  Date        : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)
    print(f"\n[instruction]\n{instruction.strip()}\n")

    system_msg = (
        "You are LabAI's benchmark orchestrator for Universidad Austral. "
        "Your job is to run MMLU benchmarks on LLMs, analyze results, and generate research reports. "
        "IMPORTANT: Always follow this exact order:\n"
        "  1. Call list_available_models to get the exact model keys you can use.\n"
        "  2. Call list_available_subjects to discover which MMLU subjects exist "
        "and find the ones relevant to the instruction.\n"
        "  3. Only then call run_benchmark using the exact model keys and subject names "
        "returned by those tools — never guess or infer them from the instruction.\n"
        "  4. After all benchmarks finish, call get_results to review rankings.\n"
        "  5. Call get_subject_breakdown for any model that needs deeper analysis.\n"
        "  6. Summarize findings and finish — a PDF report will be generated automatically."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": instruction},
    ]

    step = 0
    while True:
        step += 1
        print(f"\n[agent step {step}]")

        response = await litellm.acompletion(
            model=model_id,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2048,
            temperature=0,
        )

        msg = response.choices[0].message
        finish = response.choices[0].finish_reason

        # Show any text the agent produces
        if msg.content:
            print(f"\n[agent]\n{msg.content}")

        # Agent finished
        if finish == "stop" or finish == "end_turn":
            print("\n[agent done]")
            break

        # Agent wants to call tools
        if finish == "tool_calls" and msg.tool_calls:
            # Add assistant message with tool calls
            messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Execute each tool and collect results
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments or "{}")

                print(f"\n  --> {fn_name}({json.dumps(fn_args, ensure_ascii=False)})")
                output = await handle_tool(fn_name, fn_args, session)
                preview = output if len(output) < 300 else output[:300] + "..."
                print(f"  <-- {preview}")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      output,
                })
            continue

        # Unexpected stop
        print(f"[agent] stopped: {finish}")
        break

    # Final summary + guaranteed report
    if session["results"]:
        print_summary(list(session["results"].values()))

        # Always generate PDF, regardless of whether the agent called generate_report
        max_q = max((len(r.judge_scores) for r in session["results"].values()), default=0)
        pdf_path = generate_pdf(
            list(session["results"].values()),
            session["run_id"],
            DEFAULT_JUDGE,
            max_q,
        )
        print(f"\n  PDF report: {pdf_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LabAI Orchestrator Agent - Level 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--instruction", "-i",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Natural language instruction for the agent",
    )
    p.add_argument(
        "--orchestrator", "-o",
        choices=list(ORCHESTRATORS.keys()),
        default=DEFAULT_ORCHESTRATOR,
        help=f"Orchestrator model (default: {DEFAULT_ORCHESTRATOR})",
    )
    return p.parse_args()


async def main():
    args = parse_args()

    if not os.getenv("BRAINTRUST_API_KEY"):
        print("ERROR: BRAINTRUST_API_KEY not set.")
        sys.exit(1)

    await run_agent(args.instruction, args.orchestrator)


if __name__ == "__main__":
    asyncio.run(main())
