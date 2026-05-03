"""
LabAI - Extensible LLM Agent Evaluation Framework
Universidad Austral | AI Department

Quick start:
    from labai import Registry, AgentEvalRunner
    from labai.core.types import EvalItem, AgentResult, EvalScore
"""

from labai.core.registry import Registry
from labai.core.runner import AgentEvalRunner

__all__ = ["Registry", "AgentEvalRunner"]
