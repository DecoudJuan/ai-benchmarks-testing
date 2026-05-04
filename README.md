# LabAI — LLM Benchmark Platform

**Universidad Austral** | AI Department

Open-source LLM evaluation system that benchmarks models in two modes:

1. **MMLU** — pure knowledge across 57 subjects (~14k multiple-choice questions)
2. **Agent Eval** — agentic reasoning with real finance tools, scored by an LLM-as-judge

Results are logged to [Braintrust](https://www.braintrust.dev) and exported as interactive HTML and clean PDF reports.

---

## Architecture

```
Level 1 — Direct Benchmark (mmlu_benchmark.py)
  Fixed models, fixed dataset, fixed samples. Simplest entry point.

Level 2 — Orchestrator Agent (agent_benchmark.py)
  Claude receives a natural language instruction and autonomously
  decides which models/subjects/samples to run, then generates a report.

Level 3 — Extensible Agent Evaluation Framework (eval_agents.py + labai/)
  Evaluates tool-calling agents on domain-specific datasets.
  Plugin-based: add datasets, tools, agents, or scorers with a decorator.
```

### labai/ package

```
labai/
  core/
    types.py       # EvalItem, AgentResult, EvalScore, RunResult
    base.py        # BaseDataset, BaseTool, BaseAgent, BaseScorer (ABCs)
    registry.py    # @Registry.dataset / .tool / .agent / .scorer
    runner.py      # AgentEvalRunner — eval loop + Braintrust logging
  datasets/
    finance.py     # FinanceDataset — live Q&A generated from Yahoo Finance
    mmlu.py        # MMLUDataset — HuggingFace MMLU wrapper
  tools/
    finance.py     # get_stock_price, get_financial_ratios, calculate_return,
                   # compare_companies — all backed by live Yahoo Finance data
  agents/
    llm_agent.py   # LLMAgent — iterative tool-calling agent via litellm
                   # Retry on transient errors + XML tool-call fallback
  scorers/
    llm_judge.py   # LLMJudgeScorer — answer + reasoning + efficiency
  reports/
    html.py        # Interactive dark-mode HTML report (multi-model tabs)
    pdf.py         # Clean white PDF report
```

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```env
BRAINTRUST_API_KEY=your_key_here
OPENAI_API_KEY=sk-proj-...
OPENROUTER_API_KEY=sk-or-v1-...
ANTHROPIC_API_KEY=sk-ant-...   # optional, needed for direct Anthropic calls
```

---

## Level 1 — MMLU Benchmark

Fixed script for deterministic runs. The simplest entry point.

### How it works

1. Questions loaded from MMLU, sent to each model with a reasoning prompt
2. Model explains reasoning and gives a final answer (`ANSWER: X`)
3. LLM judge (default `gpt-4o-mini`) scores 0.0–1.0 on correctness + reasoning quality
4. Results logged to Braintrust; PDF saved to `reports/mmlu_HH-MM_YY-MM-DD.pdf`

### Quick start

```bash
# Default: gpt-4o-mini, claude-haiku, gemini-flash, qwq-32b — 100 questions each
python mmlu_benchmark.py

# Choose models
python mmlu_benchmark.py --models gpt-4o claude-sonnet deepseek-v3

# Filter by subject
python mmlu_benchmark.py --subjects math physics --samples 200

# Stronger judge
python mmlu_benchmark.py --judge openai/gpt-4o

# List available options
python mmlu_benchmark.py --list-models
python mmlu_benchmark.py --list-subjects
```

Questions are sampled **evenly across all 57 subjects** so every subject is always represented.

---

## Level 2 — Orchestrator Agent

Claude acts as an orchestrator: receives a natural language instruction, autonomously runs benchmarks, interprets results, and generates a report.

### Quick start

```bash
# Default instruction: benchmark 4 models, 50 questions each
python agent_benchmark.py

# Custom instruction
python agent_benchmark.py --instruction "Compare gpt-4o and deepseek-v3 on math and physics, 100 questions each."

# Change orchestrator model
python agent_benchmark.py --orchestrator claude-opus
python agent_benchmark.py --orchestrator gpt-4o
```

### Orchestrator tools

| Tool | Description |
|------|-------------|
| `run_benchmark` | Run MMLU eval on a model |
| `get_results` | Get all results so far, ranked by judge score |
| `get_subject_breakdown` | Top 5 / bottom 5 subjects for a model |
| `generate_report` | Save PDF report with all results |
| `list_available_models` | Show all available models |
| `list_available_subjects` | Show all 57 MMLU subjects |

---

## Level 3 — Agent Evaluation Framework

Evaluates tool-calling agents on the live finance dataset using an LLM-as-judge.

### Quick start

```bash
# Default: evaluate gpt-4o-mini and claude-haiku on finance questions
python eval_agents.py

# Evaluate multiple models
python eval_agents.py --models gpt-4o claude-haiku deepseek-v3

# Filter by category
python eval_agents.py --models gpt-4o --categories valuation returns --samples 10

# Use MMLU dataset instead
python eval_agents.py --dataset mmlu --samples 50

# Skip Braintrust logging
python eval_agents.py --no-braintrust

# List all registered components
python eval_agents.py --list
```

### Finance dataset

Questions and expected answers are **generated at load time from live Yahoo Finance data** (via `yfinance`). This ensures expected answers always match what the agent tools will return.

Tickers: `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`, `NVDA`, `META`, `NFLX`, `JPM`, `BRK-B`

| Category | Difficulty | Example |
|----------|------------|---------|
| `stock_price` | easy | Current price and daily change for AAPL |
| `valuation` | easy/medium | Trailing P/E for NVDA; compare AAPL vs AMZN P/E |
| `returns` | medium/hard | Total and annualized return for 100 AAPL shares bought 1 year ago |
| `fundamentals` | easy/medium | JPM dividend yield; META debt-to-equity |
| `comparison` | medium/hard | ROE ranking: AAPL vs MSFT vs GOOGL |
| `multi_step` | hard | Implied EPS from price + P/E; target price for 15% return |

### Finance tools

All tools call **live Yahoo Finance data** — no mocked responses.

| Tool | Description |
|------|-------------|
| `get_stock_price` | Live price + daily % change for any ticker |
| `get_financial_ratios` | Trailing P/E, Forward P/E, P/B, ROE, D/E, dividend yield |
| `calculate_return` | Total and annualized return given buy price, current price, shares |
| `compare_companies` | Side-by-side price + ratios for 2–5 tickers |

### Scoring

| Score | Weight | Description |
|-------|--------|-------------|
| `answer_score` | 60% | Correctness of the final answer (LLM judge) |
| `reasoning_score` | 30% | Quality of chain-of-thought (LLM judge) |
| `efficiency_score` | 10% | Tool-use efficiency — fewer calls = better |

### Reports

Two reports generated after every run:

- **HTML** (`reports/agent_benchmark_<model>_HH-MM_YY-MM-DD.html`) — interactive dark-mode, per-item detail, tools used, agent reasoning, costs
- **PDF** (`reports/agent_benchmark_<model>_HH-MM_YY-MM-DD.pdf`) — clean white PDF with summary, per-item table, cost breakdown, and reasoning samples

When multiple models are benchmarked, a **comparison HTML** is also generated (`reports/agent_benchmark_compare_HH-MM_YY-MM-DD.html`).

---

## Braintrust logging

Every item logged with:

| Field | Content |
|-------|---------|
| `input / output / expected` | Question, agent answer, reference answer |
| `scores` | `answer`, `reasoning`, `efficiency`, `overall` |
| `metadata` | Agent tokens (prompt + completion), agent cost USD, agent latency ms, tool calls (name + args + result), judge cost USD, judge latency ms, total cost USD |
| `tags` | Category, difficulty, `ok` / `error` |

View and compare experiments at [braintrust.dev/app](https://www.braintrust.dev/app).

---

## Resilience features

- **Auto-retry on transient errors** — LLMAgent retries OpenRouter JSON/network errors up to 3 times with exponential backoff (1 s → 2 s → 4 s). Auth and quota errors fail immediately.
- **XML tool-call fallback** — Some models (Claude via certain OpenRouter configurations) emit tool calls as Anthropic XML in the response content instead of the standard JSON `tool_calls` field. The agent detects and handles both formats automatically.

---

## Available models

| Key | Model | Provider |
|-----|-------|----------|
| `claude-haiku` | claude-3.5-haiku | Anthropic / OpenRouter |
| `claude-sonnet` | claude-sonnet-4-5 | Anthropic / OpenRouter |
| `claude-opus` | claude-opus-4-5 | Anthropic / OpenRouter |
| `gpt-4o-mini` | gpt-4o-mini | OpenAI |
| `gpt-4o` | gpt-4o | OpenAI |
| `gemini-flash` | gemini-2.0-flash-001 | Google / OpenRouter |
| `gemini-pro` | gemini-2.5-pro-preview | Google / OpenRouter |
| `llama-3.1-8b` | llama-3.1-8b-instruct | Meta / OpenRouter |
| `llama-3.3-70b` | llama-3.3-70b-instruct | Meta / OpenRouter |
| `deepseek-v3` | deepseek-chat-v3-0324 | DeepSeek / OpenRouter |
| `deepseek-r1` | deepseek-r1 | DeepSeek / OpenRouter |
| `qwq-32b` | qwq-32b | Qwen / OpenRouter |
| `mistral-nemo` | mistral-nemo | Mistral / OpenRouter |
| `phi-4` | phi-4 | Microsoft / OpenRouter |

Run `python test_models.py` to verify availability.

---

## Extend the framework

The framework is fully plugin-based — add new components without touching existing files.

### New dataset

```python
# labai/datasets/medical.py
from labai.core.base import BaseDataset
from labai.core.registry import Registry
from labai.core.types import EvalItem

@Registry.dataset("medical_qa")
class MedicalDataset(BaseDataset):
    name = "medical_qa"
    domain = "medical"
    language = "en"

    def load(self, subjects=None, n_samples=None) -> list[EvalItem]:
        ...
```

### New tool

```python
# labai/tools/medical.py
from labai.core.base import BaseTool
from labai.core.registry import Registry

@Registry.tool("drug_lookup")
class DrugLookupTool(BaseTool):
    name = "drug_lookup"
    description = "Look up information about a drug by name."

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"drug_name": {"type": "string"}},
                    "required": ["drug_name"],
                },
            },
        }

    async def execute(self, drug_name: str, **_) -> str:
        ...
```

### New agent

```python
# labai/agents/rag_agent.py
from labai.core.base import BaseAgent
from labai.core.registry import Registry
from labai.core.types import AgentResult

@Registry.agent("rag_agent")
class RAGAgent(BaseAgent):
    name = "rag-agent"
    model_id = "openai/gpt-4o"

    @property
    def tools(self): return []

    async def run(self, task: str, **_) -> AgentResult:
        ...
```

Import your new module in `eval_agents.py` and the registry picks it up automatically.

---

## Key findings

- **DeepSeek-V3** — best quality/cost ratio for knowledge tasks (MMLU)
- **Gemini Flash** — fastest and cheapest for high-volume runs
- **Claude Sonnet/Opus** — best tool-use reasoning in agentic tasks
- **Judge cost** (`gpt-4o-mini`) adds ~$0.0002–0.0005/item — relevant for large runs
- **Agent latency** is dominated by tool call count, not model speed

See [Findings.md](Findings.md) for the full write-up (Spanish + English).
