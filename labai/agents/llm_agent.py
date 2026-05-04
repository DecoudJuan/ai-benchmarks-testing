"""
LLMAgent — a tool-calling LLM agent built on litellm.

Architecture:
  - Single LLM with function-calling (OpenAI tool-use format)
  - Iterative loop: model calls tools until it produces a final text answer
  - Configurable system prompt and max iterations
  - Tracks all tool calls, token usage, latency, and USD cost

Registered as: "llm_agent"

Usage:
    from labai.agents.llm_agent import LLMAgent
    from labai.tools.finance import StockPriceTool, FinancialRatiosTool

    agent = LLMAgent(
        name="finance-agent-gpt4o",
        model_id="openai/gpt-4o",
        tools=[StockPriceTool(), FinancialRatiosTool()],
        verbose=True,
    )
    result = await agent.run("What is Apple's P/E ratio?")
"""

from __future__ import annotations

import asyncio
import json
import re
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

# ANSI colour helpers (degrade gracefully on terminals that don't support them)
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_GREY   = "\033[90m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


_MAX_RETRIES = 3  # retries on transient OpenRouter/JSON errors

# Regex patterns for Anthropic XML tool-call format that some models emit
# instead of the standard JSON tool_calls field.
_XML_FC_TAG    = "function_calls"
_XML_FC_RE     = re.compile(rf"<(?:antml:)?{_XML_FC_TAG}>(.*?)</(?:antml:)?{_XML_FC_TAG}>", re.DOTALL)
_XML_INV_RE    = re.compile(r'<(?:antml:)?invoke\s+name="([^"]+)">(.*?)</(?:antml:)?invoke>', re.DOTALL)
_XML_PARAM_RE  = re.compile(r'<(?:antml:)?parameter\s+name="([^"]+)">(.*?)</(?:antml:)?parameter>', re.DOTALL)


def _safe_cost(response: Any) -> float:
    """Extract USD cost from a litellm response without raising."""
    try:
        return litellm.completion_cost(completion_response=response) or 0.0
    except Exception:
        return 0.0


def _parse_xml_tool_calls(content: str) -> list[dict] | None:
    """
    Parse Anthropic XML tool-call blocks from response content.

    Some models (Claude via OpenRouter with certain configurations) emit tool
    calls as XML instead of the standard tool_calls JSON field. This function
    extracts them and returns a list of synthetic tool-call dicts.
    Returns None if no XML tool calls are found.
    """
    fc_match = _XML_FC_RE.search(content)
    if not fc_match:
        return None

    calls = []
    for idx, inv in enumerate(_XML_INV_RE.finditer(fc_match.group(1))):
        fn_name = inv.group(1).strip()
        params: dict[str, Any] = {}
        for p in _XML_PARAM_RE.finditer(inv.group(2)):
            raw = p.group(2).strip()
            # Try to deserialize JSON values (lists, numbers, booleans)
            try:
                params[p.group(1)] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                params[p.group(1)] = raw
        calls.append({"id": f"xml_tc_{idx}", "name": fn_name, "arguments": params})

    return calls or None


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
        verbose:        Print each tool call + result in real time (default: False).
    """

    def __init__(
        self,
        name:           str,
        model_id:       str,
        tools:          list[BaseTool],
        system_prompt:  str   = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int   = 8,
        temperature:    float = 0.0,
        max_tokens:     int   = 1024,
        verbose:        bool  = False,
    ) -> None:
        self.name           = name
        self.model_id       = model_id
        self._tools         = tools
        self.system_prompt  = system_prompt
        self.max_iterations = max_iterations
        self.temperature    = temperature
        self.max_tokens     = max_tokens
        self.verbose        = verbose

        # Build tool map for fast dispatch
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    # ── BaseAgent interface ────────────────────────────────────────────────────

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    async def run(self, task: str, item_label: str = "") -> AgentResult:
        """
        Run the agent on a single task string.

        Args:
            task:       Natural language task.
            item_label: Optional label printed in verbose output (e.g. item ID).

        Returns AgentResult with final answer, all tool calls, tokens, cost, and latency.
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
        total_cost        = 0.0
        final_output      = ""
        call_index        = 0  # sequential counter for verbose display

        for iteration in range(self.max_iterations):
            try:
                response = await self._call_llm(
                    messages    = messages,
                    tools       = tool_schemas if tool_schemas else None,
                    tool_choice = "auto" if tool_schemas else None,
                )
            except Exception as exc:
                latency_ms = (time.monotonic() - t_start) * 1_000
                if self.verbose:
                    print(f"      {_YELLOW}[error]{_RESET} {exc}")
                return AgentResult(
                    output            = "",
                    tool_calls        = all_tool_calls,
                    prompt_tokens     = prompt_tokens,
                    completion_tokens = completion_tokens,
                    total_tokens      = prompt_tokens + completion_tokens,
                    total_cost        = total_cost,
                    latency_ms        = latency_ms,
                    error             = str(exc),
                )

            # Accumulate token usage and cost
            usage = response.usage
            if usage:
                prompt_tokens     += getattr(usage, "prompt_tokens",     0) or 0
                completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            total_cost += _safe_cost(response)

            choice = response.choices[0]
            msg    = choice.message
            finish = choice.finish_reason
            content = msg.content or ""

            # Some models (Claude via OpenRouter) emit XML tool calls in the
            # content instead of the tool_calls field. Detect and normalize.
            xml_calls = _parse_xml_tool_calls(content) if content else None

            # Final text answer — no tool calls in either format
            if (finish in ("stop", "end_turn") or not getattr(msg, "tool_calls", None)) and not xml_calls:
                # Strip any XML block from the final answer if present
                final_output = _XML_FC_RE.sub("", content).strip()
                break

            # Process tool calls — prefer native tool_calls, fall back to XML
            if msg.tool_calls or xml_calls:
                if msg.tool_calls:
                    # Standard JSON tool_calls path
                    messages.append({
                        "role":       "assistant",
                        "content":    content,
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

                    # Execute tools — sequentially when verbose, concurrently otherwise
                    if self.verbose:
                        tool_results = []
                        for tc in msg.tool_calls:
                            call_index += 1
                            args_str = _fmt_args(tc.function.arguments)
                            print(
                                f"      {_CYAN}[{call_index}] --> "
                                f"{_BOLD}{tc.function.name}{_RESET}"
                                f"{_CYAN}({args_str}){_RESET}"
                            )
                            t_tool = time.monotonic()
                            result = await self._execute_tool(tc)
                            elapsed = (time.monotonic() - t_tool) * 1_000
                            preview = result.replace("\n", " | ")[:120]
                            print(
                                f"           {_GREEN}<-- {preview}{_RESET} "
                                f"{_GREY}[{elapsed:.0f}ms]{_RESET}"
                            )
                            tool_results.append(result)
                    else:
                        raw_results = await asyncio.gather(
                            *[self._execute_tool(tc) for tc in msg.tool_calls],
                            return_exceptions=True,
                        )
                        tool_results = [
                            r if isinstance(r, str) else f"Error: {r}"
                            for r in raw_results
                        ]

                    for tc_raw, output_str in zip(msg.tool_calls, tool_results):
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
                    # XML tool_calls fallback path — model emitted XML in content
                    # Add assistant message without tool_calls (model returned text)
                    messages.append({"role": "assistant", "content": content})

                    tool_result_parts: list[str] = []
                    for xml_tc in xml_calls:
                        call_index += 1
                        fn_name = xml_tc["name"]
                        fn_args = xml_tc["arguments"]
                        if self.verbose:
                            print(
                                f"      {_CYAN}[{call_index}] --> "
                                f"{_BOLD}{fn_name}{_RESET}"
                                f"{_CYAN}({fn_args}){_RESET} "
                                f"{_YELLOW}[xml format]{_RESET}"
                            )
                        t_tool = time.monotonic()
                        tool_obj = self._tool_map.get(fn_name)
                        if tool_obj:
                            try:
                                output_str = await tool_obj.execute(**fn_args)
                            except Exception as exc:
                                output_str = f"Error executing '{fn_name}': {exc}"
                        else:
                            output_str = f"Error: tool '{fn_name}' not found."

                        if self.verbose:
                            elapsed = (time.monotonic() - t_tool) * 1_000
                            preview = output_str.replace("\n", " | ")[:120]
                            print(
                                f"           {_GREEN}<-- {preview}{_RESET} "
                                f"{_GREY}[{elapsed:.0f}ms]{_RESET}"
                            )

                        all_tool_calls.append(
                            ToolCall(name=fn_name, arguments=fn_args, result=output_str)
                        )
                        tool_result_parts.append(f"[{fn_name}]\n{output_str}")

                    # Inject tool results as a user message (XML models don't use tool roles)
                    messages.append({
                        "role":    "user",
                        "content": "Tool results:\n\n" + "\n\n".join(tool_result_parts),
                    })

        else:
            # Hit max iterations — ask for a final answer
            messages.append({
                "role":    "user",
                "content": "Summarize your findings and give your final answer now.",
            })
            try:
                response = await self._call_llm(messages=messages)
                usage = response.usage
                if usage:
                    prompt_tokens     += getattr(usage, "prompt_tokens",     0) or 0
                    completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                total_cost += _safe_cost(response)
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
            total_cost        = total_cost,
            latency_ms        = latency_ms,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    async def _call_llm(self, messages: list, tools: list | None = None, tool_choice: str | None = None) -> Any:
        """
        Call litellm with automatic retry on transient OpenRouter/JSON errors.
        Retries up to _MAX_RETRIES times with exponential backoff (1s, 2s, 4s).
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await litellm.acompletion(
                    model       = self.model_id,
                    messages    = messages,
                    tools       = tools,
                    tool_choice = tool_choice,
                    temperature = self.temperature,
                    max_tokens  = self.max_tokens,
                )
            except Exception as exc:
                last_exc = exc
                err_str  = str(exc).lower()
                # Only retry on transient JSON/network errors, not auth/quota errors
                transient = any(k in err_str for k in ("json", "expecting value", "connection", "timeout", "502", "503", "504"))
                if transient and attempt < _MAX_RETRIES - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    if self.verbose:
                        print(f"      {_YELLOW}[retry {attempt + 1}/{_MAX_RETRIES - 1}]{_RESET} {exc} — waiting {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise
        raise last_exc  # unreachable but satisfies type checker

    async def _execute_tool(self, tc: Any) -> str:
        """Execute one tool call and return its string result."""
        fn_name = tc.function.name
        try:
            fn_args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            fn_args = {}

        tool = self._tool_map.get(fn_name)
        if not tool:
            return f"Error: tool '{fn_name}' not found."

        try:
            return await tool.execute(**fn_args)
        except Exception as exc:
            return f"Error executing '{fn_name}': {exc}"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_args(arguments_json: str) -> str:
    """Format tool arguments compactly for console display."""
    try:
        args = json.loads(arguments_json or "{}")
        parts = []
        for k, v in args.items():
            val = json.dumps(v) if isinstance(v, (list, dict)) else repr(v)
            parts.append(f"{k}={val}")
        return ", ".join(parts)
    except Exception:
        return arguments_json[:80]
