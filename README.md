# MMLU Benchmark with Braintrust

Evaluates LLMs on the [MMLU dataset](https://huggingface.co/datasets/cais/mmlu) (57 subjects, ~14k questions) and logs results to [Braintrust](https://www.braintrust.dev).

Tracks per-model: **accuracy**, **token usage** (prompt / completion / total), and **pairwise similarity** between models.

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:
```
BRAINTRUST_API_KEY=your_key_here   # https://www.braintrust.dev/app/settings?tab=api-keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
OPENROUTER_API_KEY=sk-or-v1-...
```

---

## Running benchmarks via CLI

### Quick start
```bash
# Default: gpt-4o-mini + claude-haiku + llama-3.1-8b, 100 questions sampled evenly across all 57 subjects
python mmlu_benchmark.py
```

### Choose models
```bash
# One model
python mmlu_benchmark.py --models gpt-4o

# Multiple models (compared side by side in Braintrust)
python mmlu_benchmark.py --models gpt-4o claude-sonnet deepseek-v3 llama-3.3-70b
```

### Change the number of questions
```bash
# 50 questions (faster, ~2 per subject)
python mmlu_benchmark.py --samples 50

# 570 questions (10 per subject)
python mmlu_benchmark.py --samples 570

# All questions in MMLU (~14k, slow)
python mmlu_benchmark.py --samples 14042
```

> Questions are sampled **evenly across all 57 subjects** so every subject is always represented.

### Filter by subject
```bash
# Single subject
python mmlu_benchmark.py --subjects math

# Multiple subjects (partial match works)
python mmlu_benchmark.py --subjects math physics computer

# All questions from a subject (no --samples limit)
python mmlu_benchmark.py --subjects math --samples 14042
```

### Combine options
```bash
python mmlu_benchmark.py --models gpt-4o claude-sonnet --subjects math physics --samples 200
```

### Explore available options
```bash
# List all available models
python mmlu_benchmark.py --list-models

# List all 57 MMLU subjects
python mmlu_benchmark.py --list-subjects
```

---

## Modifying via code

### Add or swap a model (`mmlu_benchmark.py` line 35)

Add any entry to the `MODELS` dict. Format: `"provider/model-name"`.

```python
MODELS: dict[str, str] = {
    # ... existing models ...
    "gemini-2-flash":  "openrouter/google/gemini-2.0-flash-001",
    "llama-4-scout":   "openrouter/meta-llama/llama-4-scout",
    "my-local-model":  "ollama/llama3",   # local via Ollama
}
```

Browse all OpenRouter models at [openrouter.ai/models](https://openrouter.ai/models).

### Change the default models (line 57)

```python
DEFAULT_MODELS = ["gpt-4o", "claude-sonnet", "deepseek-v3"]
```

### Swap the dataset (line 101)

Replace the `load_dataset(...)` call inside `load_mmlu()` with any HuggingFace dataset.
You only need to adjust how `input`, `expected`, and `metadata` are extracted.

```python
# HellaSwag — commonsense reasoning
ds = load_dataset("Rowan/hellaswag", split="validation")
# map: input = ctx + endings, expected = correct ending index

# TruthfulQA — factual accuracy
ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

# HumanEval — coding
ds = load_dataset("openai/openai_humaneval", split="test")
```

---

## Available models

| Key | Model ID | Provider |
|-----|----------|----------|
| `claude-haiku` | claude-3-5-haiku-20241022 | Anthropic |
| `claude-sonnet` | claude-3-5-sonnet-20241022 | Anthropic |
| `claude-opus` | claude-3-opus-20240229 | Anthropic |
| `gpt-4o-mini` | gpt-4o-mini | OpenAI |
| `gpt-4o` | gpt-4o | OpenAI |
| `llama-3.1-8b` | llama-3.1-8b-instruct:free | OpenRouter — Meta |
| `llama-3.1-70b` | llama-3.1-70b-instruct | OpenRouter — Meta |
| `llama-3.3-70b` | llama-3.3-70b-instruct | OpenRouter — Meta |
| `gemini-flash` | gemini-2.0-flash-001 | OpenRouter — Google |
| `gemini-pro` | gemini-pro-1.5 | OpenRouter — Google |
| `mistral-7b` | mistral-7b-instruct | OpenRouter — Mistral |
| `mixtral-8x7b` | mixtral-8x7b-instruct | OpenRouter — Mistral |
| `qwen-2.5-72b` | qwen-2.5-72b-instruct:nitro | OpenRouter — Alibaba |
| `deepseek-r1` | deepseek-r1 | OpenRouter — DeepSeek |
| `deepseek-v3` | deepseek-chat-v3-0324 | OpenRouter — DeepSeek |
| `phi-4` | phi-4 | OpenRouter — Microsoft |

---

## What gets logged to Braintrust

Each model run creates an experiment in the **MMLU Benchmark** project with:

- **accuracy** — % of correct answers per question
- **prompt_tokens / completion_tokens / tokens** — usage per LLM call
- **subject** — MMLU subject per question (filterable in the UI)
- **model / model_id** — experiment metadata

After all models finish, the terminal also prints:
- Token usage totals per model
- Pairwise similarity matrix (% of questions where two models gave the same answer)

View and compare experiments at [braintrust.dev/app](https://www.braintrust.dev/app).
