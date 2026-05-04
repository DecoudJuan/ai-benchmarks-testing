"""
AgentEvalRunner — orchestrates a full evaluation run.

Flow for each EvalItem:
    1. agent.run(item.input)  -> AgentResult
    2. scorer.score(item, result) -> EvalScore
    3. Collect EvalRecord(item, result, score)
    4. Log to Braintrust (optional)

Returns a RunResult with all records + aggregate metrics.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Callable

import braintrust

from labai.core.base import BaseAgent, BaseDataset, BaseScorer
from labai.core.types import AgentResult, EvalItem, EvalRecord, EvalScore, RunResult


# Progress callback type: called after each item with (index, total, record)
ProgressCallback = Callable[[int, int, EvalRecord], None]


class AgentEvalRunner:
    """
    Runs a BaseAgent against a BaseDataset, scored by a BaseScorer.

    Args:
        agent:       Agent under evaluation.
        dataset:     Dataset providing EvalItems.
        scorer:      Scorer producing EvalScore for each item.
        braintrust_project: If set, results are logged to this Braintrust project.
        concurrency: Max parallel evaluations (default: 5).
        on_progress: Optional callback after each completed item.

    Example:
        runner = AgentEvalRunner(
            agent=my_agent,
            dataset=FinanceDataset(),
            scorer=LLMJudgeScorer(),
            braintrust_project="finance-agent-eval",
        )
        result = await runner.run()
        print(result.avg_overall)
    """

    def __init__(
        self,
        agent:               BaseAgent,
        dataset:             BaseDataset,
        scorer:              BaseScorer,
        braintrust_project:  str | None = None,
        concurrency:         int        = 5,
        on_progress:         ProgressCallback | None = None,
        subjects:            list[str] | None = None,
        n_samples:           int | None       = None,
    ) -> None:
        self.agent              = agent
        self.dataset            = dataset
        self.scorer             = scorer
        self.braintrust_project = braintrust_project
        self.concurrency        = concurrency
        self.on_progress        = on_progress
        self.subjects           = subjects
        self.n_samples          = n_samples

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(self) -> RunResult:
        """Load items, evaluate all, return RunResult."""
        run_id = str(uuid.uuid4())[:8]
        items  = self.dataset.load(subjects=self.subjects, n_samples=self.n_samples)

        if not items:
            raise ValueError(f"Dataset '{self.dataset.name}' returned 0 items.")

        print(f"\n[runner] run_id={run_id}  agent={self.agent.name}  "
              f"dataset={self.dataset.name}  items={len(items)}")

        # Optionally open a Braintrust experiment
        bt_experiment = None
        if self.braintrust_project:
            bt_experiment = braintrust.init(
                project=self.braintrust_project,
                experiment=f"{self.agent.name}-{run_id}",
            )

        records: list[EvalRecord] = []
        n_width   = len(str(len(items)))
        verbose   = getattr(self.agent, "verbose", False)

        # When verbose, run sequentially so tool traces don't interleave
        effective_concurrency = 1 if verbose else self.concurrency
        semaphore = asyncio.Semaphore(effective_concurrency)

        async def eval_one(index: int, item: EvalItem) -> EvalRecord:
            async with semaphore:
                prefix = f"  [{index+1:>{n_width}}/{len(items)}]"
                if verbose:
                    q_preview = item.input.split("\n")[0][:70]
                    print(f"\n{prefix} {item.id}  \033[90m{q_preview}\033[0m")

                record = await self._eval_item(item, bt_experiment)

                if self.on_progress:
                    self.on_progress(index + 1, len(items), record)
                else:
                    score     = record.score
                    n_tools   = len(record.result.tool_calls)
                    err_flag  = " \033[91m[ERR]\033[0m" if record.result.error else ""
                    if score.overall >= 0.75:
                        colour = "\033[92m"
                    elif score.overall >= 0.50:
                        colour = "\033[93m"
                    else:
                        colour = "\033[91m"
                    reset = "\033[0m"

                    if verbose:
                        tool_names = " -> ".join(tc.name for tc in record.result.tool_calls) or "(no tools)"
                        print(
                            f"      tools: {tool_names}\n"
                            f"      score: {colour}overall={score.overall:.2f}{reset}  "
                            f"answer={score.answer_score:.2f}  "
                            f"reasoning={score.reasoning_score:.2f}  "
                            f"efficiency={score.efficiency_score:.2f}  "
                            f"({n_tools} calls  {record.result.latency_ms:.0f}ms){err_flag}"
                        )
                    else:
                        print(
                            f"{prefix} {item.id:<20} "
                            f"{colour}overall={score.overall:.2f}{reset}  "
                            f"answer={score.answer_score:.2f}  "
                            f"tools={n_tools}{err_flag}"
                        )
                return record

        tasks   = [eval_one(i, item) for i, item in enumerate(items)]
        records = await asyncio.gather(*tasks)

        if bt_experiment:
            bt_experiment.flush()

        return RunResult(
            run_id       = run_id,
            agent_name   = self.agent.name,
            dataset_name = self.dataset.name,
            records      = list(records),
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _eval_item(
        self,
        item:          EvalItem,
        bt_experiment: object | None,
    ) -> EvalRecord:
        """Evaluate a single item: run agent -> score -> log."""

        # Run agent
        t0 = time.monotonic()
        try:
            result: AgentResult = await self.agent.run(item.input, item_label=item.id)
        except TypeError:
            result = await self.agent.run(item.input)
        except Exception as exc:
            result = AgentResult(output="", error=str(exc))

        elapsed_ms = (time.monotonic() - t0) * 1_000

        # Score
        try:
            score: EvalScore = await self.scorer.score(item, result)
        except Exception as exc:
            score = EvalScore(details={"error": str(exc)})

        record = EvalRecord(item=item, result=result, score=score)

        # Log to Braintrust
        if bt_experiment is not None:
            self._log_to_braintrust(bt_experiment, item, result, score, elapsed_ms)

        return record

    @staticmethod
    def _log_to_braintrust(
        experiment: object,
        item:       EvalItem,
        result:     AgentResult,
        score:      EvalScore,
        latency_ms: float,
    ) -> None:
        try:
            # Build tool call metadata
            tool_calls_meta = [
                {
                    "name":      tc.name,
                    "arguments": tc.arguments,
                    "result":    tc.result[:500],
                }
                for tc in result.tool_calls
            ]

            experiment.log(
                input    = item.input,
                output   = result.output,
                expected = item.expected,
                scores   = {
                    **score.to_dict(),
                    # Surface individual judge sub-scores at top level for Braintrust charts
                    "answer":     score.answer_score,
                    "reasoning":  score.reasoning_score,
                    "efficiency": score.efficiency_score,
                },
                metadata = {
                    # Item metadata
                    **item.metadata,
                    # Agent metrics
                    "agent_prompt_tokens":     result.prompt_tokens,
                    "agent_completion_tokens": result.completion_tokens,
                    "agent_total_tokens":      result.total_tokens,
                    "agent_cost_usd":          round(result.total_cost, 6),
                    "agent_latency_ms":        round(latency_ms, 1),
                    "agent_tool_calls":        len(result.tool_calls),
                    "tool_calls":              tool_calls_meta,
                    "agent_error":             result.error or None,
                    # Judge metrics
                    "judge_model":             score.details.get("judge_model", ""),
                    "judge_answer_score":      score.answer_score,
                    "judge_reasoning_score":   score.reasoning_score,
                    "judge_efficiency_score":  score.efficiency_score,
                    "judge_overall_score":     score.overall,
                    "judge_rationale":         score.details.get("rationale", ""),
                    "judge_cost_usd":          round(score.details.get("judge_cost", 0.0), 6),
                    "judge_latency_ms":        score.details.get("judge_latency_ms", 0.0),
                    "judge_prompt_tokens":     score.details.get("judge_prompt_tok", 0),
                    "judge_completion_tokens": score.details.get("judge_completion_tok", 0),
                    "judge_total_tokens":      score.details.get("judge_total_tok", 0),
                    # Totals
                    "total_cost_usd":          round(result.total_cost + score.details.get("judge_cost", 0.0), 6),
                    "total_tokens":            result.total_tokens + score.details.get("judge_total_tok", 0),
                },
                tags = [
                    item.metadata.get("category", ""),
                    item.metadata.get("difficulty", ""),
                    "error" if result.error else "ok",
                ],
            )
        except Exception:
            pass  # Never let logging break the eval
