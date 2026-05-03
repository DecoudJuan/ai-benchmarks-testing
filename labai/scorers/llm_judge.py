"""
LLMJudgeScorer — evaluates agent responses using an LLM as judge.

Produces three independent scores (all 0.0–1.0):

  answer_score     — correctness of the final answer vs expected
  reasoning_score  — quality of reasoning / chain-of-thought
  efficiency_score — tool-use efficiency (fewer calls = better)

The three scores map to EvalScore, whose .overall property applies
a 60/30/10 weighted average.

Registered as: "llm_judge"

Rubric (sent to the judge model):
  answer_score:
    1.0  — fully correct, no meaningful errors
    0.7  — mostly correct, minor inaccuracy or incomplete
    0.4  — partially correct, significant error but shows understanding
    0.0  — incorrect, irrelevant, or completely wrong

  reasoning_score:
    1.0  — clear, logical, well-structured reasoning
    0.7  — adequate reasoning with minor gaps
    0.4  — partial reasoning, jumps or weak logic
    0.0  — no reasoning, or reasoning contradicts the answer

  efficiency_score (automatic, not from LLM):
    Based on number of tool calls vs an expected-minimum baseline.
    0 tool calls when tools were needed  -> 0.0
    1 tool call                          -> 1.0
    Each extra call beyond minimum       -> -0.15 penalty (floor 0.0)
"""

from __future__ import annotations

import json
import re

import litellm

from labai.core.base import BaseScorer
from labai.core.registry import Registry
from labai.core.types import AgentResult, EvalItem, EvalScore

litellm.set_verbose       = False
litellm.suppress_debug_info = True

# ── Judge prompt ───────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """You are an expert evaluator for AI agent responses.

## Task
Evaluate the agent's response against the expected answer.

## Question
{question}

## Expected answer
{expected}

## Agent's response
{response}

## Tool calls made ({n_tools} total)
{tool_summary}

## Scoring rubric

answer_score (0.0 - 1.0):
  1.0  — fully correct, matches expected answer precisely
  0.7  — mostly correct, minor inaccuracy or incomplete
  0.4  — partially correct, significant error but partial understanding
  0.0  — incorrect or completely off

reasoning_score (0.0 - 1.0):
  1.0  — clear, logical, well-structured reasoning
  0.7  — adequate reasoning with minor gaps
  0.4  — some reasoning but with jumps or weak logic
  0.0  — no reasoning, or contradicts the answer

## Instructions
- Focus on factual accuracy for answer_score.
- Focus on the chain of thought and logical flow for reasoning_score.
- Be strict but fair.
- Respond ONLY with valid JSON, no markdown, no explanation outside the JSON.

## Response format
{{
  "answer_score": <float 0.0-1.0>,
  "reasoning_score": <float 0.0-1.0>,
  "rationale": "<one or two sentences explaining your scores>"
}}"""


# ── Scorer ─────────────────────────────────────────────────────────────────────

@Registry.scorer("llm_judge")
class LLMJudgeScorer(BaseScorer):
    """
    LLM-as-judge scorer.

    Args:
        judge_model:   litellm model string for the judge (default: gpt-4o-mini).
        min_tool_calls: Expected minimum tool calls for a task requiring tools.
                        Used to compute efficiency_score. (default: 1)
    """

    name = "llm_judge"

    def __init__(
        self,
        judge_model:   str = "openai/gpt-4o-mini",
        min_tool_calls: int = 1,
    ) -> None:
        self.judge_model    = judge_model
        self.min_tool_calls = min_tool_calls

    # ── BaseScorer interface ───────────────────────────────────────────────────

    async def score(self, item: EvalItem, result: AgentResult) -> EvalScore:
        """
        Evaluate one agent result against the expected answer.

        Returns EvalScore with answer, reasoning, and efficiency scores.
        """
        # Build tool summary for the judge
        tool_lines = (
            "\n".join(
                f"  - {tc.name}({json.dumps(tc.arguments)}) -> {tc.result[:120]}..."
                if len(tc.result) > 120 else
                f"  - {tc.name}({json.dumps(tc.arguments)}) -> {tc.result}"
                for tc in result.tool_calls
            )
            if result.tool_calls
            else "  (no tool calls)"
        )

        prompt = _JUDGE_PROMPT.format(
            question     = item.input,
            expected     = item.expected,
            response     = result.output or "(empty)",
            n_tools      = len(result.tool_calls),
            tool_summary = tool_lines,
        )

        # Call judge
        answer_score   = 0.0
        reasoning_score = 0.0
        rationale       = ""

        try:
            response = await litellm.acompletion(
                model       = self.judge_model,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0,
                max_tokens  = 256,
            )
            raw = response.choices[0].message.content or ""
            answer_score, reasoning_score, rationale = _parse_judge_response(raw)
        except Exception as exc:
            rationale = f"Judge error: {exc}"

        # Efficiency score (deterministic, not from LLM)
        efficiency_score = _compute_efficiency(
            n_tool_calls   = len(result.tool_calls),
            min_tool_calls = self.min_tool_calls,
            has_error      = bool(result.error),
        )

        return EvalScore(
            answer_score     = answer_score,
            reasoning_score  = reasoning_score,
            efficiency_score = efficiency_score,
            details          = {
                "rationale":   rationale,
                "n_tool_calls": len(result.tool_calls),
                "judge_model": self.judge_model,
                "agent_error": result.error or None,
            },
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_judge_response(raw: str) -> tuple[float, float, str]:
    """Extract scores from the judge JSON response. Returns (answer, reasoning, rationale)."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        data = json.loads(cleaned)
        answer   = float(data.get("answer_score",   0.0))
        reasoning = float(data.get("reasoning_score", 0.0))
        rationale = str(data.get("rationale", ""))
        # Clamp to [0, 1]
        answer    = max(0.0, min(1.0, answer))
        reasoning = max(0.0, min(1.0, reasoning))
        return answer, reasoning, rationale
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback: try regex
        a_match = re.search(r'"answer_score"\s*:\s*([\d.]+)', cleaned)
        r_match = re.search(r'"reasoning_score"\s*:\s*([\d.]+)', cleaned)
        answer   = float(a_match.group(1)) if a_match else 0.0
        reasoning = float(r_match.group(1)) if r_match else 0.0
        return answer, reasoning, f"Parse fallback. Raw: {raw[:120]}"


def _compute_efficiency(
    n_tool_calls:   int,
    min_tool_calls: int,
    has_error:      bool,
) -> float:
    """
    Compute efficiency score based on number of tool calls.

    Logic:
      - Agent crashed (error) -> 0.0
      - 0 calls when min > 0  -> 0.0 (couldn't use tools)
      - Exactly min calls     -> 1.0 (optimal)
      - Each extra call       -> -0.15 penalty
    """
    if has_error:
        return 0.0
    if n_tool_calls == 0 and min_tool_calls > 0:
        return 0.0
    if n_tool_calls <= min_tool_calls:
        return 1.0
    extra   = n_tool_calls - min_tool_calls
    penalty = extra * 0.15
    return max(0.0, 1.0 - penalty)
