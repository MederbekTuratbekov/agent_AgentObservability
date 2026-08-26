"""
Финальная сборка проекта — соединяем три слоя наблюдаемости вместе.

Берём агента (например, из отдельного проекта с LangGraph), оборачиваем в:
  - trace logging (видим каждый шаг)
  - token budget проверку (не превышаем лимит контекста)
  - LLM-as-judge оценку (проверяем качество финальных ответов)

Это уже близко к тому, как выглядит production-ready
мини-агент, а не просто учебный скрипт.
"""

import time

from trace_logger import TraceLogger
from token_budget import count_messages_tokens, apply_sliding_window
from llm_judge import judge_answer


def run_agent_with_observability(question: str, reference_answer: str, run_agent_fn):
    """
    Оборачивает вызов агента наблюдаемостью:
        1. Проверяем token budget перед вызовом
        2. Логируем через trace_logger
        3. После получения ответа — оцениваем через LLM-judge

    run_agent_fn — функция агента, принимает question, возвращает ответ строкой.

    Примечание: sliding window здесь считает бюджет и обрезает messages,
    но сам run_agent_fn принимает только question (строку), а не список
    messages — поэтому обрезанный контекст пока не передаётся в вызов
    агента напрямую. Если твой агент умеет принимать историю сообщений,
    передавай в него именно `messages`, а не только `question`.
    """
    tracer = TraceLogger(f"agent_run_{int(time.time())}")

    messages = [{"role": "user", "content": question}]
    token_count = count_messages_tokens(messages)

    print(f"Вопрос: {question}")
    print(f"Токенов в запросе: {token_count}")

    if token_count > 4000:
        print("Превышен token budget — применяем sliding window")
        messages = apply_sliding_window(messages, max_tokens=4000)

    with tracer.span("llm_call", "agent", input_data=question) as record:
        answer = run_agent_fn(question)
        record(answer)

    tracer.print_summary()
    trace_path = tracer.save()

    print("\nОценка через LLM-as-judge...")
    verdict = judge_answer(question, answer, reference_answer)
    print(f"Вердикт: {verdict.verdict.value}")
    print(f"Обоснование: {verdict.reasoning}")

    return {
        "question": question,
        "answer": answer,
        "verdict": verdict.verdict.value,
        "reasoning": verdict.reasoning,
        "token_count": token_count,
        "trace_path": trace_path,
    }