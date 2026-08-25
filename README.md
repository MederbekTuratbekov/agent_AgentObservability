# Eval + Context + Debug

Часть 5 из 5 — финал серии проектов по блоку `agent_junior`.

## Problem

Агент из проектов 2–4 работает, но без наблюдаемости:
не видно сколько токенов уходит в контекст, что происходит
на каждом шаге, и насколько хорош финальный ответ по смыслу
(не по точному совпадению строк, как в проекте 1).

## Approach

- **Token budget**: подсчёт токенов через `tiktoken`,
  `sliding_window` обрезает старую историю, сохраняя system message
  и самые свежие сообщения
- **Trace logging**: `TraceLogger` в стиле Langfuse — записывает
  каждый LLM/tool вызов со временем выполнения, сохраняет в JSON
- **LLM-as-judge**: отдельная модель оценивает ответы агента
  по вердиктам correct / partially_correct / incorrect —
  нужно там, где точное сравнение строк (как в проекте 1) не работает
- `full_pipeline.py` оборачивает вызов агента всеми тремя слоями сразу

## Results

`full_pipeline.py` выводит: число токенов в запросе, полный трейс
шагов агента, вердикт LLM-судьи с обоснованием.

## How to run

```bash
pip install -r requirements.txt

# Windows PowerShell:
$env:OPENAI_API_KEY="твой_ключ"

python token_budget.py     # демо sliding window
python trace_logger.py      # демо trace logging
python llm_judge.py          # демо LLM-as-judge
python full_pipeline.py       # всё вместе (с mock-агентом по умолчанию)
```

Чтобы подключить реального агента из проекта 3 вместо mock:
в `full_pipeline.py` замени `mock_agent` на `run_agent`
из `langgraph_memory/graph.py`.

## Структура проекта

```
agent-observability/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── main.py
    ├── token_budget.py     # подсчёт токенов + sliding window
    ├── trace_logger.py       # TraceLogger — запись шагов агента
    ├── llm_judge.py             # LLM-as-judge оценка качества ответов
    └── full_pipeline.py           # сборка всех трёх слоёв вместе
```

## Итог блока agent_junior

Все 5 тем закрыты пятью проектами:

| Проект | Темы agent_junior |
|---|---|
| 1. llm_basics_prompting | llm_basics, prompting |
| 2. tool_calling_react | tool_calling (core), ReAct pattern |
| 3. langgraph_memory | LangGraph базовый, agent memory |
| 4. rag_basics | rag_basics — полностью |
| 5. eval_context_debug | context, eval_debug |

Что не вошло явно ни в один проект: multi-agent basics
(orchestrator-worker концептуально) — это следующий логичный шаг
после закрытия agent_junior, естественный переход к более
сложным темам.