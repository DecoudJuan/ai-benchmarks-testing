# LabAI - LLM Benchmark Platform

**Universidad Austral** | AI Department

Evaluates LLMs on the [MMLU dataset](https://huggingface.co/datasets/cais/mmlu) (57 subjects, ~14k questions).

Each answer is scored by an **LLM-as-judge** (correctness + reasoning quality), results are logged to [Braintrust](https://www.braintrust.dev), and a **PDF report** is generated at the end.

---

## Architecture

```
Level 1 - Orchestrator Agent (agent_benchmark.py)
  Claude receives a natural language instruction and autonomously
  decides which models/subjects/samples to run, interprets results,
  and calls the benchmark tools.

Level 2 - Direct Benchmark Script (mmlu_benchmark.py)
  Deterministic script: fixed models, fixed dataset, fixed samples.
  Used directly via CLI or called by the Level 1 agent.
```

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:
```
BRAINTRUST_API_KEY=your_key_here   # https://www.braintrust.dev/app/settings?tab=api-keys
OPENAI_API_KEY=sk-proj-...
OPENROUTER_API_KEY=sk-or-v1-...
```

---

## Level 1 - Orchestrator Agent

Claude acts as an orchestrator: receives a natural language instruction and autonomously runs benchmarks, interprets results, and generates the report.

### Quick start
```bash
# Default instruction: benchmark 4 models, 50 questions each, generate report
python agent_benchmark.py

# Custom instruction
python agent_benchmark.py --instruction "Compare gpt-4o and deepseek-v3 on math and physics, 100 questions each. Identify the weakest subject for each model."

# Change orchestrator model
python agent_benchmark.py --orchestrator claude-opus
python agent_benchmark.py --orchestrator gpt-4o
```

### Available orchestrators
| Key | Model |
|-----|-------|
| `claude-haiku` | claude-3.5-haiku (fast, cheap) |
| `claude-sonnet` | claude-sonnet-4-5 (default, balanced) |
| `claude-opus` | claude-opus-4-5 (most capable) |
| `gpt-4o-mini` | gpt-4o-mini (cheap) |
| `gpt-4o` | gpt-4o |

### What the agent can do (tools)
| Tool | Description |
|------|-------------|
| `run_benchmark` | Run MMLU eval on a model (model, subjects, samples) |
| `get_results` | Get all results so far, ranked by judge score |
| `get_subject_breakdown` | Top 5 / bottom 5 subjects for a model |
| `generate_report` | Save PDF report with all results |
| `list_available_models` | Show all 16 available models |
| `list_available_subjects` | Show all 57 MMLU subjects |

---

## Level 2 - Direct Benchmark Script

Fixed script for deterministic runs. Also used internally by the agent.

### How it works

1. Questions are loaded from MMLU and sent to each model
2. Each model is asked to **explain its reasoning** and then give its final answer (`ANSWER: X`)
3. An **LLM judge** (default: `gpt-4o-mini`) scores each response from 0.0 to 1.0:
   - `1.0` — correct answer + clear, accurate reasoning
   - `0.7` — correct answer + weak or missing reasoning
   - `0.4` — wrong answer but shows partial understanding
   - `0.0` — wrong answer with no relevant reasoning
4. Results are logged to Braintrust (judge score, accuracy, tokens per call)
5. A **PDF report** is saved to `reports/mmlu_<run_id>.pdf`

---

## Running benchmarks via CLI

### Quick start
```bash
# Default: gpt-4o-mini, claude-haiku, gemini-flash, qwen-2.5-72b — 100 questions each
python mmlu_benchmark.py
```

### Choose models
```bash
python mmlu_benchmark.py --models gpt-4o claude-sonnet deepseek-v3
```

### Change the judge model
```bash
# Use a stronger judge for more accurate scoring
python mmlu_benchmark.py --judge openai/gpt-4o

# Use a cheaper judge for speed
python mmlu_benchmark.py --judge openai/gpt-4o-mini
```

### Change the number of questions
```bash
# 50 questions (~2 per subject, fast)
python mmlu_benchmark.py --samples 50

# 570 questions (10 per subject)
python mmlu_benchmark.py --samples 570

# All questions in MMLU (~14k)
python mmlu_benchmark.py --samples 14042
```

> Questions are sampled **evenly across all 57 subjects** so every subject is always represented.

### Filter by subject
```bash
python mmlu_benchmark.py --subjects math
python mmlu_benchmark.py --subjects math physics computer
python mmlu_benchmark.py --subjects math --samples 14042   # all math questions
```

### Combine options
```bash
python mmlu_benchmark.py --models gpt-4o claude-sonnet --subjects math physics --samples 200 --judge openai/gpt-4o
```

### Explore available options
```bash
python mmlu_benchmark.py --list-models
python mmlu_benchmark.py --list-subjects
```

---

## PDF Report

After every run a PDF is saved to `reports/mmlu_<run_id>.pdf` with:

| Section | Content |
|---------|---------|
| **Page 1 — Leaderboard** | All models ranked by judge score, with accuracy and token usage |
| **Per-model pages** | Judge score, accuracy, token totals, full subject breakdown table, top 5 / bottom 5 subjects |
| **Last page** | Horizontal bar chart comparing judge scores across all models |

Scores are color-coded: green ≥ 75%, yellow ≥ 50%, red < 50%.

---

## What gets logged to Braintrust

Each model run creates an experiment in the **MMLU Benchmark** project:

- **judge** — LLM-as-judge score (0.0–1.0) per question
- **accuracy** — binary correct/incorrect per question
- **prompt_tokens / completion_tokens / tokens** — per LLM call
- **subject** — MMLU subject (filterable in the UI)
- **model / model_id / judge** — experiment metadata

View and compare at [braintrust.dev/app](https://www.braintrust.dev/app).

---

## Modifying via code

### Add a model
Edit the `MODELS` dict at the top of `mmlu_benchmark.py`:

```python
MODELS: dict[str, str] = {
    # ... existing models ...
    "gemini-2-flash":  "openrouter/google/gemini-2.0-flash-001",
    "llama-4-scout":   "openrouter/meta-llama/llama-4-scout",
    "my-local-model":  "ollama/llama3",
}
```

Browse all OpenRouter models at [openrouter.ai/models](https://openrouter.ai/models).

### Change the default models
```python
DEFAULT_MODELS = ["gpt-4o", "claude-sonnet", "deepseek-v3"]
```

### Swap the dataset
Replace the `load_dataset(...)` call inside `load_mmlu()` with any HuggingFace dataset.
Adjust how `input`, `expected`, and `metadata` are extracted to match the new schema.

```python
# HellaSwag — commonsense reasoning
ds = load_dataset("Rowan/hellaswag", split="validation")

# TruthfulQA — factual accuracy
ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

# HumanEval — coding
ds = load_dataset("openai/openai_humaneval", split="test")
```

### Change the judge scoring criteria
Edit the `JUDGE_PROMPT` string in `mmlu_benchmark.py` to adjust the rubric.

---

## Available models

| Key | Model | Provider |
|-----|-------|----------|
| `claude-haiku` | claude-3.5-haiku | OpenRouter — Anthropic |
| `claude-sonnet` | claude-sonnet-4-5 | OpenRouter — Anthropic |
| `claude-opus` | claude-opus-4-5 | OpenRouter — Anthropic |
| `gpt-4o-mini` | gpt-4o-mini | OpenAI |
| `gpt-4o` | gpt-4o | OpenAI |
| `llama-3.1-8b` | llama-3.1-8b-instruct | OpenRouter — Meta |
| `llama-3.1-70b` | llama-3.1-70b-instruct | OpenRouter — Meta |
| `llama-3.3-70b` | llama-3.3-70b-instruct | OpenRouter — Meta |
| `gemini-flash` | gemini-2.0-flash-001 | OpenRouter — Google |
| `gemini-pro` | gemini-2.5-pro-preview | OpenRouter — Google |
| `mistral-nemo` | mistral-nemo | OpenRouter — Mistral |
| `mixtral-8x7b` | mixtral-8x7b-instruct | OpenRouter — Mistral |
| `qwen-2.5-72b` | qwen-2.5-72b-instruct:nitro | OpenRouter — Alibaba |
| `deepseek-r1` | deepseek-r1 | OpenRouter — DeepSeek |
| `deepseek-v3` | deepseek-chat-v3-0324 | OpenRouter — DeepSeek |
| `phi-4` | phi-4 | OpenRouter — Microsoft |

All 16 models verified working. Run `python test_models.py` to re-check availability.
