# LabAI — Sistema de Benchmarking de Modelos de Lenguaje

> **Proyecto:** Universidad Austral — Departamento de Inteligencia Artificial  
> **Repositorio:** `ai-benchmarks`

---

## Resumen del proyecto

LabAI es un sistema open-source de evaluación de modelos de lenguaje grande (LLMs) construido desde cero. Permite comparar modelos en dos modalidades: (1) conocimiento puro mediante el benchmark MMLU y (2) capacidad de razonamiento agéntico con herramientas reales, usando un juez LLM para la evaluación automática. Los resultados se registran en Braintrust para trazabilidad y se exportan como reportes HTML y PDF.

---

## Arquitectura del sistema

```
ai-benchmarks/
├── labai/
│   ├── core/
│   │   ├── types.py       # Tipos de datos compartidos (EvalItem, AgentResult, RunResult…)
│   │   ├── base.py        # Clases base abstractas (BaseAgent, BaseDataset, BaseScorer)
│   │   ├── registry.py    # Registro de componentes via decoradores
│   │   └── runner.py      # AgentEvalRunner: orquesta evaluación + logging Braintrust
│   ├── agents/
│   │   └── llm_agent.py   # LLMAgent: agente con tool-calling sobre litellm
│   ├── datasets/
│   │   ├── finance.py     # Dataset de preguntas financieras con categorías/dificultad
│   │   └── mmlu.py        # Wrapper del dataset MMLU (HuggingFace)
│   ├── scorers/
│   │   └── llm_judge.py   # LLMJudgeScorer: juez LLM con rubrica (answer/reasoning/efficiency)
│   ├── tools/
│   │   └── finance.py     # Herramientas financieras (precio acción, ratios, retornos, comparación)
│   └── reports/
│       ├── html.py        # Reporte HTML interactivo multi-modelo con tabs
│       └── pdf.py         # Reporte PDF blanco con métricas y razonamiento
├── mmlu_benchmark.py      # Runner del benchmark MMLU (standalone)
├── eval_agents.py         # Runner del agent benchmark
└── Findings.md            # Este archivo
```

---

## Stack tecnológico

| Componente | Tecnología | Propósito |
|---|---|---|
| Llamadas a LLMs | [litellm](https://github.com/BerriAI/litellm) | Interfaz unificada para 100+ modelos vía un API |
| Routing de modelos | [OpenRouter](https://openrouter.ai) | Acceso a Qwen, Llama, Mistral, DeepSeek, Gemini |
| APIs directas | OpenAI, Anthropic | GPT-4o y Claude via SDK oficial |
| Observabilidad | [Braintrust](https://www.braintrust.dev) | Logging de experimentos, spans, scores, costos |
| Dataset MMLU | [HuggingFace datasets](https://huggingface.co/datasets/cais/mmlu) | 57 materias, ~14k preguntas de opción múltiple |
| Generación PDF | [fpdf2](https://pychoicesfpdf2.readthedocs.io) | Reportes PDF con fondo blanco |
| Cálculo de costos | litellm `completion_cost()` | Costo en USD por llamada a modelo |
| Async | asyncio + litellm async | Evaluación concurrente de ítems |

---

## Modelos evaluados

| Alias | Modelo | Proveedor |
|---|---|---|
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
| `qwen-2.5-72b` | qwen-2.5-72b-instruct | Qwen / OpenRouter |
| `mistral-nemo` | mistral-nemo | Mistral / OpenRouter |
| `phi-4` | phi-4 | Microsoft / OpenRouter |

---

## Metodología de evaluación

### Benchmark MMLU
- **Dataset:** 57 materias de conocimiento (STEM, humanidades, ciencias sociales, etc.)
- **Formato:** Preguntas de opción múltiple (A/B/C/D)
- **Métricas:**
  - **Accuracy**: respuesta correcta (determinista)
  - **Judge Score**: evaluación por LLM-as-judge (considera razonamiento)
- **Juez:** `gpt-4o-mini` por defecto — puntúa de 0.0 a 1.0 según calidad de razonamiento
- **Logging:** Braintrust via `braintrust.Eval` — incluye tokens, costo agente y costo juez

### Agent Benchmark
- **Dataset:** Preguntas de finanzas reales (valoración, retornos, comparaciones)
- **Agente:** `LLMAgent` con tool-calling iterativo (hasta 8 rondas)
- **Herramientas disponibles:** `get_stock_price`, `get_financial_ratios`, `calculate_return`, `compare_companies`
- **Scoring (LLM-as-Judge):**
  - `answer_score` × 0.60 — correctitud de la respuesta
  - `reasoning_score` × 0.30 — calidad del razonamiento
  - `efficiency_score` × 0.10 — uso eficiente de herramientas
- **Logging Braintrust:** Por ítem — tokens agente + tokens juez, costo agente + costo juez, latencia, tool calls, rationale del juez, tags por categoría/dificultad
- **Reportes generados:** HTML interactivo (dark mode, multi-tab) + PDF blanco

---

## Datos registrados en Braintrust

Por cada ítem evaluado se loguean:

- Input, output esperado, output del agente
- Scores: `answer`, `reasoning`, `efficiency`, `overall`
- Tokens: agente (prompt + completion) y juez
- Costos USD: agente, juez, total
- Latencia: agente y juez por separado
- Tool calls: nombre, argumentos, resultado (primeros 500 chars)
- Rationale del juez
- Tags: categoría, dificultad, ok/error

---

## Convención de nombres de archivos

```
agent_benchmark_<modelo>_YY-MM-DD_HH-MM.html   # reporte HTML del agent benchmark
agent_benchmark_<modelo>_YY-MM-DD_HH-MM.pdf    # reporte PDF del agent benchmark
agent_benchmark_compare_YY-MM-DD_HH-MM.html    # comparación multi-modelo
mmlu_YY-MM-DD_HH-MM.pdf                        # reporte PDF del MMLU benchmark
```

---

## Hallazgos

- **DeepSeek-V3** obtiene resultados consistentemente altos en MMLU con costo muy bajo por token, posicionándolo como la mejor relación calidad/precio en tareas de conocimiento.
- **Gemini Flash** sobresale en velocidad y costo, siendo ideal para benchmarks con muchos ítems.
- **Claude Sonnet/Opus** lidera en razonamiento agéntico gracias a mejor seguimiento de instrucciones y uso de herramientas.
- **Qwen-2.5-72b** presentó problemas de disponibilidad por proveedor (Novita vía OpenRouter con `:nitro`); solucionado removiendo el sufijo de routing.
- El juez LLM (`gpt-4o-mini`) agrega ~$0.0002-0.0005 por ítem — relevante para runs grandes.
- La latencia del agente es dominada por el número de tool calls, no por el modelo.

---

---
---

# LabAI — LLM Benchmarking System

> **Project:** Universidad Austral — AI Department  
> **Repository:** `ai-benchmarks`

---

## Project Summary

LabAI is an open-source LLM evaluation system built from scratch. It benchmarks models in two modes: (1) pure knowledge via MMLU and (2) agentic reasoning with real tools, using an LLM-as-judge for automatic evaluation. Results are logged to Braintrust for observability and exported as interactive HTML and clean PDF reports.

---

## Architecture

The system follows a registry-based plugin pattern: datasets, agents, scorers, and tools are registered via decorators and discovered at runtime. The runner orchestrates evaluation with configurable concurrency, collects metrics, logs to Braintrust, and generates reports.

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| LLM calls | litellm | Unified interface for 100+ models |
| Model routing | OpenRouter | Access to Qwen, Llama, Mistral, DeepSeek, Gemini |
| Direct APIs | OpenAI, Anthropic | GPT-4o and Claude |
| Observability | Braintrust | Experiment tracking, spans, scores, costs |
| MMLU dataset | HuggingFace datasets | 57 subjects, ~14k multiple-choice questions |
| PDF generation | fpdf2 | White-background PDF reports |
| Cost tracking | litellm `completion_cost()` | Per-call USD cost for all models |
| Async | asyncio + litellm async | Concurrent item evaluation |

---

## Evaluation Methodology

### MMLU Benchmark
- 57 knowledge domains, multiple-choice format (A/B/C/D)
- Metrics: rule-based accuracy + LLM judge score (0.0–1.0)
- Logs tokens, cost (agent + judge), and subject per item to Braintrust

### Agent Benchmark
- Finance QA dataset with tool-calling agent
- Tools: stock price, financial ratios, return calculation, company comparison
- Weighted scoring: answer×0.60 + reasoning×0.30 + efficiency×0.10
- Full Braintrust logging: tokens, costs, latencies, tool calls, judge rationale, tags

---

## Key Findings

- **DeepSeek-V3**: best quality/cost ratio for knowledge tasks
- **Gemini Flash**: fastest and cheapest for high-volume runs
- **Claude Sonnet/Opus**: best tool-use reasoning in agentic tasks
- Judge cost (`gpt-4o-mini`) adds ~$0.0002–0.0005/item — relevant for large runs
- Agent latency is dominated by tool call count, not model speed

---

