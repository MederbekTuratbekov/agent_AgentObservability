# Agent Observability

Три слоя наблюдаемости для LLM-агентов, собранные вручную (в стиле Langfuse/LangSmith), без внешних SaaS: **trace logging**, **token budget / sliding window** и **LLM-as-judge** оценка качества ответов.

## Место в линейке проектов

Пятый проект серии — здесь агент (из [проекта 3 на LangGraph](https://github.com/MederbekTuratbekov/agent_LanggraphAgentMemory.git)) оборачивается инфраструктурой, без которой любой агент — чёрный ящик: непонятно, где он ошибся, сколько стоил вызов и насколько хорош был ответ.

## Зачем три отдельных слоя, а не один

| Слой | Отвечает на вопрос |
|---|---|
| **Trace logging** | Что именно произошло на каждом шаге и сколько это заняло? |
| **Token budget** | Не упрёмся ли в лимит контекста модели? |
| **LLM-as-judge** | Ответ реально правильный — или просто похож на правильный? |

Каждый слой решает свою проблему независимо и может использоваться отдельно от двух других.

## Trace logging

Записывает последовательность событий (LLM-вызовы, tool-вызовы) с временными метками — без этого невозможно понять, где агент ошибся: на этапе рассуждения, выбора инструмента или интерпретации результата.

```python
tracer = TraceLogger("agent_run")

with tracer.span("llm_call", "agent", input_data=question) as record:
    answer = run_agent_fn(question)
    record(answer)

tracer.print_summary()
tracer.save()  # -> agent_run_<timestamp>.json
```

`span()` — context manager, сам замеряет время выполнения блока и пишет событие в трейс. Каждый трейс сохраняется в JSON для последующего анализа — вход, выход (обрезанные до 500 символов) и длительность каждого шага.

## Token budget + sliding window

У любой модели есть лимит контекста. Sliding window — простое решение: обрезаем историю сообщений, чтобы уложиться в бюджет, **сохраняя system message и самые свежие сообщения**, а не самые старые.

```python
token_count = count_messages_tokens(messages)

if token_count > 4000:
    messages = apply_sliding_window(messages, max_tokens=4000)
```

Подсчёт токенов — через `tiktoken` с кодировкой `o200k_base` (актуальна для GPT-4o / GPT-4o-mini; `cl100k_base` даёт неточный результат на этих моделях). Обрезка идёт с начала истории — свежий контекст обычно важнее для ответа на текущий вопрос, чем самое старое сообщение диалога.

## LLM-as-judge

Точное сравнение строк не работает, когда ответ агента — целое предложение, а не одно слово: `"Париж"` и `"Столица Франции — Париж"` формально разные строки, но оба правильные. LLM-судья читает вопрос, ответ и эталон, и оценивает **по смыслу**.

```python
verdict = judge_answer(question, agent_answer, reference_answer)
# verdict.verdict    -> "correct" | "partially_correct" | "incorrect"
# verdict.reasoning  -> краткое объяснение вердикта
```

Судья работает на `temperature=0.0` — для оценочной функции важна стабильность, а не креативность. Есть batch-версия (`judge_batch`) — прогоняет список ответов и считает `correct_rate` по всей выборке.

## Полная сборка (`full_pipeline.py`)

```
question
   │
   ▼
count_messages_tokens ──► превышен бюджет? ──► apply_sliding_window
   │
   ▼
TraceLogger.span("llm_call") ──► run_agent_fn(question)
   │
   ▼
judge_answer(question, answer, reference_answer)
   │
   ▼
{question, answer, verdict, reasoning, token_count, trace_path}
```

`run_agent_with_observability()` — точка входа, принимает любую функцию агента (`run_agent_fn: str -> str`) и оборачивает её всеми тремя слоями. По умолчанию в `main.py` используется `mock_agent` — заглушка, чтобы проверить пайплайн без Redis/pgvector; для реального прогона подставляется `run_agent` из проекта 3.

## Структура проекта

```
agent-observability/
├── src/
│   ├── main.py               # точка входа, пример с mock_agent
│   ├── full_pipeline.py        # сборка всех трёх слоёв вместе
│   ├── trace_logger.py           # TraceLogger — запись шагов агента
│   ├── token_budget.py             # подсчёт токенов + sliding window
│   └── llm_judge.py                  # LLM-as-judge оценка ответов
├── requirements.txt
├── .env.example
└── .gitignore
```

## Установка

```bash
git clone https://github.com/<username>/agent-observability.git
cd agent-observability
pip install -r requirements.txt
```

Создай `.env` на основе `.env.example`:

```
OPENAI_API_KEY=sk-...
```

## Запуск

```bash
python src/main.py
```

Прогоняет один вопрос через `mock_agent`, показывая все три слоя в работе:

```
Вопрос: Кто создал Python?
Токенов в запросе: 12

=== Trace: agent_run_1735000000 ===
Всего событий: 1
Общее время: 512.3 ms

  [llm_call] agent — 501.2ms
Трейс сохранён: agent_run_1735000000_1735000000.json

Оценка через LLM-as-judge...
Вердикт: correct
Обоснование: Ответ верно называет Гвидо ван Россума создателем Python.

=== Финальный результат ===
{'question': 'Кто создал Python?', 'answer': '...', 'verdict': 'correct', ...}
```

Чтобы подключить настоящего агента вместо заглушки — замени `mock_agent` в `main.py` на `run_agent` из `langgraph_memory/graph.py` (проект 3); потребуются также `state.py` и `tools.py` из того проекта, плюс поднятый Redis.

## Известное ограничение

`apply_sliding_window` обрезает и возвращает список `messages`, но `run_agent_fn` в текущей сборке принимает только `question` (строку) — обрезанный контекст пока не передаётся в сам вызов агента напрямую. Если агент умеет принимать историю сообщений целиком, стоит передавать в него именно `messages`, а не только `question`.

## Технологии

- **`tiktoken`** — точный подсчёт токенов под конкретную модель
- **OpenAI API** (`gpt-4o-mini`) — LLM-судья, `response_format=json_object`
- **Pydantic** — строгая схема вердикта судьи (`JudgeResult`)
- **`contextlib.contextmanager`** — замер времени через `with tracer.span(...)`

## Возможные улучшения

- [ ] Передавать обрезанный `messages` в `run_agent_fn`, а не только `question`
- [ ] Дашборд поверх сохранённых трейсов (сейчас — только консоль + JSON)
- [ ] Логирование стоимости запроса (токены × цена модели) в трейс
- [ ] Сравнение нескольких LLM-судей между собой на одном наборе ответов

## Лицензия

MIT