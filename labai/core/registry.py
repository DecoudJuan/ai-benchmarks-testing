"""
Registry — global catalog of all LabAI components.

Each component type (dataset, tool, agent, scorer) is registered once
with a decorator and then looked up by name at runtime.

Usage:
    @Registry.dataset("finance_qa")
    class FinanceDataset(BaseDataset):
        ...

    @Registry.tool("get_stock_price")
    class StockPriceTool(BaseTool):
        ...

    ds   = Registry.get_dataset("finance_qa")
    tool = Registry.get_tool("get_stock_price")
"""

from __future__ import annotations
from typing import Any, Type, TypeVar

T = TypeVar("T")


class _ComponentRegistry:
    """
    Thread-safe, type-aware component registry.
    Stores class objects (not instances) — the caller decides when to instantiate.
    """

    def __init__(self, kind: str) -> None:
        self._kind  = kind
        self._store: dict[str, type] = {}

    # ── Registration ───────────────────────────────────────────────────────────

    def register(self, name: str):
        """Decorator that registers a class under *name*."""
        def decorator(cls: type) -> type:
            if name in self._store:
                raise ValueError(
                    f"[Registry] {self._kind} '{name}' is already registered by "
                    f"{self._store[name].__qualname__}. "
                    f"Each name must be unique."
                )
            self._store[name] = cls
            # Attach the registry name to the class for easy introspection
            cls._registry_name = name
            return cls
        return decorator

    # ── Lookup ─────────────────────────────────────────────────────────────────

    def get(self, name: str) -> type:
        """Return the class registered under *name*. Raises KeyError if absent."""
        try:
            return self._store[name]
        except KeyError:
            available = ", ".join(sorted(self._store)) or "(none)"
            raise KeyError(
                f"[Registry] {self._kind} '{name}' not found. "
                f"Available: {available}"
            )

    def list(self) -> list[str]:
        """Sorted list of all registered names."""
        return sorted(self._store)

    def __repr__(self) -> str:
        return f"_ComponentRegistry({self._kind!r}, {self.list()})"


# ── Public facade ──────────────────────────────────────────────────────────────

class Registry:
    """
    Facade over all four component registries.

    Decorators (to register):
        @Registry.dataset("name")
        @Registry.tool("name")
        @Registry.agent("name")
        @Registry.scorer("name")

    Lookups (to retrieve class):
        Registry.get_dataset("name")  -> type[BaseDataset]
        Registry.get_tool("name")     -> type[BaseTool]
        Registry.get_agent("name")    -> type[BaseAgent]
        Registry.get_scorer("name")   -> type[BaseScorer]

    Catalog:
        Registry.list_datasets()  -> list[str]
        Registry.list_tools()     -> list[str]
        Registry.list_agents()    -> list[str]
        Registry.list_scorers()   -> list[str]
    """

    _datasets = _ComponentRegistry("dataset")
    _tools    = _ComponentRegistry("tool")
    _agents   = _ComponentRegistry("agent")
    _scorers  = _ComponentRegistry("scorer")

    # ── Decorators ─────────────────────────────────────────────────────────────

    @classmethod
    def dataset(cls, name: str):
        """Register a BaseDataset subclass."""
        return cls._datasets.register(name)

    @classmethod
    def tool(cls, name: str):
        """Register a BaseTool subclass."""
        return cls._tools.register(name)

    @classmethod
    def agent(cls, name: str):
        """Register a BaseAgent subclass."""
        return cls._agents.register(name)

    @classmethod
    def scorer(cls, name: str):
        """Register a BaseScorer subclass."""
        return cls._scorers.register(name)

    # ── Getters ────────────────────────────────────────────────────────────────

    @classmethod
    def get_dataset(cls, name: str) -> type:
        return cls._datasets.get(name)

    @classmethod
    def get_tool(cls, name: str) -> type:
        return cls._tools.get(name)

    @classmethod
    def get_agent(cls, name: str) -> type:
        return cls._agents.get(name)

    @classmethod
    def get_scorer(cls, name: str) -> type:
        return cls._scorers.get(name)

    # ── Catalog ────────────────────────────────────────────────────────────────

    @classmethod
    def list_datasets(cls) -> list[str]:
        return cls._datasets.list()

    @classmethod
    def list_tools(cls) -> list[str]:
        return cls._tools.list()

    @classmethod
    def list_agents(cls) -> list[str]:
        return cls._agents.list()

    @classmethod
    def list_scorers(cls) -> list[str]:
        return cls._scorers.list()

    @classmethod
    def summary(cls) -> dict[str, list[str]]:
        """Full catalog as a dict for inspection / logging."""
        return {
            "datasets": cls.list_datasets(),
            "tools":    cls.list_tools(),
            "agents":   cls.list_agents(),
            "scorers":  cls.list_scorers(),
        }
