# Arquitectura

LabAI evalúa modelos de lenguaje en tres niveles que comparten el mismo paquete `labai/`. Este documento explica cómo
encajan las piezas; el uso de cada script está en el [README](../README.md) y el análisis de resultados en
[Findings.md](../Findings.md).

## Los tres niveles

| Nivel | Script | Qué hace |
|---|---|---|
| 1 | `mmlu_benchmark.py` | Corre MMLU sobre modelos fijos, con un juez LLM. Es el punto de entrada más simple. |
| 2 | `agent_benchmark.py` | Un modelo orquestador (Claude por defecto) recibe una instrucción en lenguaje natural y decide qué modelos, materias y muestras correr, con herramientas como `run_benchmark`, `get_results` y `generate_report`. |
| 3 | `eval_agents.py` | Evalúa agentes con herramientas sobre un dataset de dominio, con un juez LLM. Todo es enchufable. |

Utilidades: `test_models.py` y `test_claude_ids.py` verifican qué ids de modelo responden realmente a través de
OpenRouter y los proveedores directos.

## El paquete labai

```
labai/core/        tipos (EvalItem, AgentResult, EvalScore, RunResult), clases base abstractas y el Registry
labai/datasets/    finance.py (preguntas generadas con datos vivos de Yahoo Finance), mmlu.py (HuggingFace)
labai/tools/       finance.py: get_stock_price, get_financial_ratios, calculate_return, compare_companies
labai/agents/      llm_agent.py: agente con tool-calling iterativo sobre litellm
labai/scorers/     llm_judge.py: juez LLM (respuesta, razonamiento, eficiencia)
labai/reports/     html.py (informe interactivo, pestañas por modelo) y pdf.py (informe blanco)
```

## Flujo de una corrida (nivel 3)

`AgentEvalRunner` (`labai/core/runner.py`) recorre los `EvalItem` de un dataset, con concurrencia acotada:

1. `agent.run(item.input)` devuelve un `AgentResult` (respuesta, llamadas a herramientas, tokens, costo, latencia).
2. `scorer.score(item, result)` devuelve un `EvalScore`.
3. Se arma un `EvalRecord` y, si se pidió, se registra en Braintrust (tags, scores, metadatos).
4. Al final se devuelve un `RunResult` con todos los registros y las métricas agregadas, y se generan los reportes.

El puntaje total pondera 60 % la respuesta, 30 % el razonamiento y 10 % la eficiencia en el uso de herramientas.

## Extensión por registro

Los componentes se registran con decoradores (`@Registry.dataset`, `.tool`, `.agent`, `.scorer`), y `eval_agents.py`
los activa al importar sus módulos. Sumar un dataset, una herramienta, un agente o un scorer no requiere tocar los
archivos existentes; los ejemplos están en la sección "Extend the framework" del README.

## Robustez del agente

`LLMAgent` reintenta hasta 3 veces con espera exponencial (1 s, 2 s, 4 s) ante errores transitorios de OpenRouter, y
entiende tanto `tool_calls` en JSON como llamadas escritas en XML de Anthropic dentro del contenido.
