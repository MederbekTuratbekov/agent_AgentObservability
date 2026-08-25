"""
Финальная сборка проекта 5 — и всего блока agent_junior.

Берём агента из проекта 3 (langgraph_memory), оборачиваем в:
  - trace logging (видим каждый шаг)
  - token budget проверку (не превышаем лимит контекста)
  - LLM-as-judge оценку (проверяем качество финальных ответов)

Это уже близко к тому, как выглядит production-ready
мини-агент, а не просто учебный скрипт.

Примечание: для полного запуска нужны файлы из проекта 3
(graph.py, state.py, tools.py) — либо скопируй их сюда,
либо запускай этот файл из объединённой папки со всеми частями.
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

    run_agent_fn — функция агента (например run_agent из graph.py проекта 3),
    принимает question, возвращает ответ строкой.
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


def demo_run():
    """
    Демонстрация без реального вызова графа (заглушка вместо LangGraph-агента),
    чтобы файл можно было проверить без поднятия Redis/pgvector.

    Для реального прогона замени mock_agent на run_agent из
    langgraph_memory/graph.py
    """
    def mock_agent(question: str) -> str:
        time.sleep(0.5)  # имитация задержки реального вызова
        return "Python был создан Гвидо ван Россумом."

    result = run_agent_with_observability(
        question="Кто создал Python?",
        reference_answer="Guido van Rossum",
        run_agent_fn=mock_agent,
    )

    print("\n=== Финальный результат ===")
    print(result)


if __name__ == "__main__":
    demo_run()