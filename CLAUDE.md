# LabAI (ai-benchmarks-testing)

Sistema open-source para comparar LLMs: MMLU (conocimiento) y evaluación de agentes con herramientas reales de
finanzas, puntuada por un juez LLM. Resultados a Braintrust y reportes HTML/PDF.

## Stack
- Python 3, asyncio, litellm (OpenAI, Anthropic y OpenRouter), Braintrust, HuggingFace `datasets`, yfinance, fpdf2.

## Comandos
```bash
pip install -r requirements.txt
cp .env.example .env                  # BRAINTRUST_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY
python mmlu_benchmark.py              # nivel 1
python agent_benchmark.py             # nivel 2: orquestador
python eval_agents.py --list          # nivel 3: lista los componentes registrados
python eval_agents.py --no-braintrust --samples 5
python test_models.py                 # qué modelos responden hoy
```
No hay suite de tests ni build: `test_models.py` y `test_claude_ids.py` son scripts que llaman a las APIs reales
(cuestan plata y necesitan las claves).

## Estructura
- `mmlu_benchmark.py`, `agent_benchmark.py`, `eval_agents.py` — los tres niveles.
- `labai/core/` tipos, bases abstractas, `Registry`, `AgentEvalRunner`.
- `labai/datasets|tools|agents|scorers|reports/` componentes enchufables y reportes.
- `docs/arquitectura.md` explica el flujo; `Findings.md` es el informe del proyecto.

## Convenciones
- Los componentes nuevos se registran con `@Registry.dataset|tool|agent|scorer` y se importan en `eval_agents.py`;
  no se modifican los existentes.
- Los datasets de finanzas se generan con datos vivos de Yahoo Finance: las respuestas esperadas coinciden con lo que
  devuelven las herramientas. No mockear.
- Nunca commitear `.env` ni `reports/` (ambos están en `.gitignore`).
- Windows: los scripts reconfiguran stdout a UTF-8.

## Git
Commits y PRs sin atribución a Claude: nada de `Co-Authored-By: Claude` ni "Generated with Claude Code".
Commits convencionales (`feat:`, `fix:`, `refactor:`).
