"""
Abstract Base Classes for all LabAI components.

To add a new dataset, tool, agent, or scorer:
  1. Subclass the appropriate ABC
  2. Implement all abstract methods
  3. Decorate with @Registry.dataset / @Registry.tool / etc.

That's it — no other files need to change.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from labai.core.types import EvalItem, AgentResult, EvalScore, ToolCall


# ── Dataset ────────────────────────────────────────────────────────────────────

class BaseDataset(ABC):
    """
    Provides evaluation items for a benchmark run.

    Extend this to add new datasets (MMLU, finance, legal, medical...).
    """
    name:     str   # e.g. "mmlu", "finance_qa"
    domain:   str   # e.g. "general", "finance"
    language: str = "en"

    @abstractmethod
    def load(
        self,
        subjects:  list[str] | None = None,
        n_samples: int | None       = None,
    ) -> list[EvalItem]:
        """
        Load and return evaluation items.

        Args:
            subjects:  Optional filter (dataset-specific, e.g. MMLU subjects).
            n_samples: Max items to return. None = all.

        Returns:
            List of EvalItem, sampled evenly across categories when limited.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, domain={self.domain!r})"


# ── Tool ───────────────────────────────────────────────────────────────────────

class BaseTool(ABC):
    """
    A single capability an agent can invoke.

    Extend this to add new tools (finance APIs, web search, calculators...).
    One class = one tool function.
    """
    name:        str   # Must match the function name in get_schema()
    description: str

    @abstractmethod
    def get_schema(self) -> dict:
        """
        Return an OpenAI-compatible function tool schema:
        {
            "type": "function",
            "function": {
                "name": ...,
                "description": ...,
                "parameters": { "type": "object", "properties": {...} }
            }
        }
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """
        Execute the tool and return a string result.
        Should never raise — catch exceptions and return an error string.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ── Agent ──────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    An AI agent that can be evaluated.

    Extend this to define different agent architectures:
    - Simple LLM (single call)
    - Tool-calling LLM (multi-step)
    - Multi-agent pipeline
    - RAG agent
    """
    name:     str   # Human-readable label for reports
    model_id: str   # LLM model identifier (litellm format)

    @property
    @abstractmethod
    def tools(self) -> list[BaseTool]:
        """Tools available to this agent."""
        ...

    @abstractmethod
    async def run(self, task: str) -> AgentResult:
        """
        Run the agent on a task and return the full result.

        Args:
            task: Natural language task description.

        Returns:
            AgentResult with final output, all tool calls, and token usage.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model_id!r})"


# ── Scorer ─────────────────────────────────────────────────────────────────────

class BaseScorer(ABC):
    """
    Evaluates how well an agent responded to an EvalItem.

    Extend this to add new scoring strategies:
    - LLM-as-judge
    - Exact match
    - Regex / structured output match
    - Human scoring interface
    """
    name: str

    @abstractmethod
    async def score(
        self,
        item:   EvalItem,
        result: AgentResult,
    ) -> EvalScore:
        """
        Produce a score for one agent result.

        Args:
            item:   The original eval item (includes expected answer).
            result: What the agent produced.

        Returns:
            EvalScore with answer, reasoning, and efficiency scores.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
