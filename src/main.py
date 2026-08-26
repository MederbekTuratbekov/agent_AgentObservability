"""
main.py — точка входа проекта agent-observability.

Здесь собираются все три слоя наблюдаемости в один пример запуска:
  - trace_logger   — логирование шагов агента
  - token_budget   — контроль лимита токенов
  - llm_judge      — оценка качества ответа через LLM-судью
  - full_pipeline  — обёртка, которая соединяет всё вместе

Для реального прогона с полноценным LangGraph-агентом (проект 3)
замени mock_agent на run_agent из langgraph_memory/graph.py —
для этого нужны файлы graph.py, state.py, tools.py из проекта 3.
"""

import time

from dotenv import load_dotenv

load_dotenv()  # подтягиваем OPENAI_API_KEY и другие ключи из .env

from full_pipeline import run_agent_with_observability


def mock_agent(question: str) -> str:
    """Заглушка вместо реального LangGraph-агента — чтобы проверить пайплайн без внешних зависимостей (Redis/pgvector)."""
    time.sleep(0.5)  # имитация задержки настоящего вызова LLM
    return "Python был создан Гвидо ван Россумом."


def main():
    result = run_agent_with_observability(
        question="Кто создал Python?",
        reference_answer="Guido van Rossum",
        run_agent_fn=mock_agent,
    )

    print("\n=== Финальный результат ===")
    print(result)


if __name__ == "__main__":
    main()