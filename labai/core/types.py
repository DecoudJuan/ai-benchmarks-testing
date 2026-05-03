"""
Core data types shared across the entire LabAI framework.
All components communicate through these types — never through concrete classes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ── Input / Output ─────────────────────────────────────────────────────────────

@dataclass
class EvalItem:
    """A single evaluation task passed to an agent."""
    id:       str
    input:    str           # Full task description
    expected: str           # Ground-truth answer
    metadata: dict = field(default_factory=dict)  # domain, difficulty, category, etc.


@dataclass
class ToolCall:
    """A single tool call made by an agent during execution."""
    name:       str
    arguments:  dict
    result:     str
    latency_ms: float = 0.0


@dataclass
class AgentResult:
    """Everything an agent produced while handling one EvalItem."""
    output:            str                     # Final answer text
    tool_calls:        list[ToolCall] = field(default_factory=list)
    reasoning:         str           = ""
    prompt_tokens:     int           = 0
    completion_tokens: int           = 0
    total_tokens:      int           = 0
    latency_ms:        float         = 0.0
    error:             str           = ""


# ── Scores ─────────────────────────────────────────────────────────────────────

@dataclass
class EvalScore:
    """Scores for a single evaluation record (all 0.0–1.0)."""
    answer_score:     float = 0.0   # Correctness of the final answer
    reasoning_score:  float = 0.0   # Quality of the reasoning process
    efficiency_score: float = 0.0   # Tool use efficiency (fewer calls = better)
    details:          dict  = field(default_factory=dict)

    @property
    def overall(self) -> float:
        """Weighted composite: answer quality matters most."""
        return (
            self.answer_score    * 0.60 +
            self.reasoning_score * 0.30 +
            self.efficiency_score * 0.10
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "answer_score":     round(self.answer_score,    3),
            "reasoning_score":  round(self.reasoning_score, 3),
            "efficiency_score": round(self.efficiency_score, 3),
            "overall":          round(self.overall,          3),
        }


# ── Run aggregation ─────────────────────────────────────────────────────────────

@dataclass
class EvalRecord:
    """Complete record for one item: input + agent output + score."""
    item:   EvalItem
    result: AgentResult
    score:  EvalScore


@dataclass
class RunResult:
    """Aggregated results for a full benchmark run."""
    run_id:       str
    agent_name:   str
    dataset_name: str
    records:      list[EvalRecord] = field(default_factory=list)

    # ── Aggregate metrics ──────────────────────────────────────────────────────

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def avg_overall(self) -> float:
        return self._mean([r.score.overall for r in self.records])

    @property
    def avg_answer(self) -> float:
        return self._mean([r.score.answer_score for r in self.records])

    @property
    def avg_reasoning(self) -> float:
        return self._mean([r.score.reasoning_score for r in self.records])

    @property
    def avg_efficiency(self) -> float:
        return self._mean([r.score.efficiency_score for r in self.records])

    @property
    def total_tokens(self) -> int:
        return sum(r.result.total_tokens for r in self.records)

    @property
    def avg_tool_calls(self) -> float:
        return self._mean([len(r.result.tool_calls) for r in self.records])

    @property
    def error_rate(self) -> float:
        return self._mean([1.0 if r.result.error else 0.0 for r in self.records])

    def scores_by_category(self) -> dict[str, float]:
        """Average overall score grouped by metadata['category']."""
        from collections import defaultdict
        groups: dict[str, list[float]] = defaultdict(list)
        for rec in self.records:
            cat = rec.item.metadata.get("category", "unknown")
            groups[cat].append(rec.score.overall)
        return {k: self._mean(v) for k, v in sorted(groups.items())}

    def scores_by_difficulty(self) -> dict[str, float]:
        """Average overall score grouped by metadata['difficulty']."""
        from collections import defaultdict
        groups: dict[str, list[float]] = defaultdict(list)
        for rec in self.records:
            diff = rec.item.metadata.get("difficulty", "unknown")
            groups[diff].append(rec.score.overall)
        return {k: self._mean(v) for k, v in sorted(groups.items())}
