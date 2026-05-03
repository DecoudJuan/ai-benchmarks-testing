# MMLU Benchmark with Braintrust

Evaluates LLMs on the [MMLU dataset](https://huggingface.co/datasets/cais/mmlu) (57 subjects, ~14k questions) and logs results to [Braintrust](https://www.braintrust.dev).

## Setup

```bash
pip install -r requirements.txt
```

Add your keys to `.env`:
```
BRAINTRUST_API_KEY=your_key_here   # https://www.braintrust.dev/app/settings?tab=api-keys
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
```

## Usage

```bash
# Default: gpt-4o-mini, claude-haiku, llama-3.1-8b — 100 questions each
python mmlu_benchmark.py

# Specific models
python mmlu_benchmark.py --models gpt-4o claude-sonnet deepseek-v3

# Filter by subject and sample size
python mmlu_benchmark.py --subjects math physics --samples 200

# List all available models / subjects
python mmlu_benchmark.py --list-models
python mmlu_benchmark.py --list-subjects
```

## Available models

| Key | Provider |
|-----|----------|
| `claude-haiku`, `claude-sonnet`, `claude-opus` | Anthropic |
| `gpt-4o-mini`, `gpt-4o` | OpenAI |
| `llama-3.1-8b`, `llama-3.1-70b`, `llama-3.3-70b` | OpenRouter (Meta) |
| `gemini-flash`, `gemini-pro` | OpenRouter (Google) |
| `mistral-7b`, `mixtral-8x7b` | OpenRouter (Mistral) |
| `qwen-2.5-72b`, `deepseek-r1`, `deepseek-v3`, `phi-4` | OpenRouter |

## Results

View experiments at https://www.braintrust.dev/app — compare accuracy across models, filter by subject, and track changes over time.
