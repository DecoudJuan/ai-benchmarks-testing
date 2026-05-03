"""
LLMAgent — a tool-calling LLM agent built on litellm.

Architecture:
  - Single LLM with function-calling (OpenAI tool-use format)
  - Iterative loop: model calls tools until it produces a final text answer
  - Configurable system prompt and max iterations
  - Tracks all tool calls, token usage, and latency

Registered as: "llm_agent"

Usage:
    from labai.agents.llm_agent import LLMAgent
    from labai.tools.finance import StockPriceTool, FinancialRatiosTool

    agent = LLMAgent(
        name="finance-agent-gpt4o",
        model_id="openai/gpt-4o",
        tools=[StockPriceTool(), FinancialRatiosTool()],
    )
    result = await agent.run("What is Apple's P/E ratio?")
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import litellm

from labai.core.base import BaseAgent, BaseTool
from labai.core.registry import Registry
from labai.core.types import AgentResult, ToolCall

litellm.set_verbose       = False
litellm.suppress_debug_info = True

DEFAULT_SYSTEM_PROMPT = (
    "You are a financial research assistant. "
    "Use the available tools to retrieve data, then provide a clear, concise answer. "
    "Always base your conclusions on the tool data you retrieve — do not guess numbers. "
    "Explain your reasoning before giving the final answer."
)


@Registry.agent("llm_agent")
class LLMAgent(BaseAgent):
    """
    General-purpose tool-calling LLM agent.

    Args:
        name:           Human-readable label (used in reports).
        model_id:       litellm model string (e.g. 'openai/gpt-4o-mini').
        tools:          List of BaseTool instances available to the agent.
        system_prompt:  System message. Defaults to finance research assistant prompt.
        max_iterations: Max tool-call rounds before forcing a final answer (default: 8).
        temperature:    Sampling temperature (default: 0 for determinism).
        max_tokens:     Max tokens for each LLM call (default: 1024).
    """

    def __init__(
        self,
        name:           str,
        model_id:       str,
        tools:          list[BaseTool],
        system_prompt:  str  = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int  = 8,
        temperature:    float = 0.0,
        max_tokens:     int  = 1024,
    ) -> None:
        self.name           = name
        self.model_id       = model_id
        self._tools         = tools
        self.system_prompt  = system_prompt
        self.max_iterations = max_iterations
        self.temperature    = temperature
        self.max_tokens     = max_tokens

        # Build tool map for fast dispatch
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    # ── BaseAgent interface ────────────────────────────────────────────────────

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    async def run(self, task: str) -> AgentResult:
        """
        Run the agent on a single task string.

        Returns AgentResult with final answer, all tool calls, tokens, and latency.
        """
        t_start = time.monotonic()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": task},
        ]

        tool_schemas = [t.get_schema() for t in self._tools]
        all_tool_calls: list[ToolCall] = []
        prompt_tokens     = 0
        completion_tokens = 0
        final_output      = ""

        for iteration in range(self.max_iterations):
            try:
                response = await litellm.acompletion(
                    model       = self.model_id,
                    messages    = messages,
                    tools       = tool_schemas if tool_schemas else None,
                    tool_choice = "auto" if tool_schemas else None,
                    temperature = self.temperature,
                    max_tokens  = self.max_tokens,
                )
            except Exception as exc:
                latency_ms = (time.monotonic() - t_start) * 1_000
                return AgentResult(
                    output            = "",
                    tool_calls        = all_tool_calls,
                    prompt_tokens     = prompt_tokens,
                    completion_tokens = completion_tokens,
                    total_tokens      = prompt_tokens + completion_tokens,
                    latency_ms        = latency_ms,
                    error             = str(exc),
                )

            # Accumulate token usage
            usage = response.usage
            if usage:
                prompt_tokens     += getattr(usage, "prompt_tokens",     0) or 0
                completion_tokens += getattr(usage, "completion_tokens", 0) or 0

            choice     = response.choices[0]
            msg        = choice.message
            finish     = choice.finish_reason

            # Final text answer
            if finish in ("stop", "end_turn") or not getattr(msg, "tool_calls", None):
                final_output = msg.content or ""
                break

            # Process tool calls
            if msg.tool_calls:
                # Add assistant message with tool calls
                messages.append({
                    "role":       "assistant",
                    "content":    msg.content or "",
                    "tool_calls": [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # Execute tools concurrently
                tool_results = await asyncio.gather(
                    *[self._execute_tool(tc) for tc in msg.tool_calls],
                    return_exceptions=True,
                )

                for tc_raw, tc_result in zip(msg.tool_calls, tool_results):
                    output_str = (
                        tc_result if isinstance(tc_result, str)
                        else f"Error: {tc_result}"
                    )
                    all_tool_calls.append(
                        ToolCall(
                            name      = tc_raw.function.name,
                            arguments = json.loads(tc_raw.function.arguments or "{}"),
                            result    = output_str,
                        )
                    )
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc_raw.id,
                        "content":      output_str,
                    })

        else:
            # Hit max iterations — ask for a final answer
            messages.append({
                "role":    "user",
                "content": "Summarize your findings and give your final answer now.",
            })
            try:
                response = await litellm.acompletion(
                    model       = self.model_id,
                    messages    = messages,
                    temperature = self.temperature,
                    max_tokens  = self.max_tokens,
                )
                usage = response.usage
                if usage:
                    prompt_tokens     += getattr(usage, "prompt_tokens",     0) or 0
                    completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                final_output = response.choices[0].message.content or ""
            except Exception:
                final_output = "Max iterations reached without a final answer."

        latency_ms = (time.monotonic() - t_start) * 1_000

        return AgentResult(
            output            = final_output,
            tool_calls        = all_tool_calls,
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            total_tokens      = prompt_tokens + completion_tokens,
            latency_ms        = latency_ms,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    async def _execute_tool(self, tc: Any) -> str:
        """Execute one tool call and return its string result."""
        fn_name = tc.function.name
        try:
            fn_args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            fn_args = {}

        t0   = time.monotonic()
        tool = self._tool_map.get(fn_name)
        if not tool:
            return f"Error: tool '{fn_name}' not found."

        try:
            result     = await tool.execute(**fn_args)
            elapsed_ms = (time.monotonic() - t0) * 1_000
            return result
        except Exception as exc:
            return f"Error executing '{fn_name}': {exc}"
