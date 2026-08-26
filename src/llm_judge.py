"""
LLM-as-judge — используем LLM для оценки качества ответов другой LLM
(или того же агента), там где точное сравнение строк не работает.

Когда это нужно:
  - если ответ агента — целое предложение, а не одно слово,
    точное сравнение строк не работает: "Париж" и "Столица Франции — Париж"
    формально разные строки, но оба правильные
  - LLM-судья читает вопрос + ответ + правильный ответ и решает:
    это по сути верно или нет
"""

import json
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()  # чтобы модуль сам находил OPENAI_API_KEY, даже если его импортировали не через main.py

client = OpenAI()
JUDGE_MODEL = "gpt-4o-mini"


class Verdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


class JudgeResult(BaseModel):
    """Результат оценки LLM-судьи для одного ответа."""
    verdict: Verdict = Field(description="Вердикт: correct / partially_correct / incorrect")
    reasoning: str = Field(description="Краткое объяснение вердикта")


JUDGE_SYSTEM_PROMPT = """Ты — судья, оценивающий качество ответов AI-агента.

Тебе дают:
- вопрос пользователя
- ответ агента
- эталонный (правильный) ответ

Оцени, насколько ответ агента соответствует эталону ПО СМЫСЛУ,
а не по точному совпадению текста. Разные формулировки одного
и того же факта — это correct, а не incorrect.

Вердикты:
- correct: ответ по сути верен, содержит правильную информацию
- partially_correct: ответ частично верен, но есть неточности или пропуски
- incorrect: ответ неверен или не отвечает на вопрос

Отвечай СТРОГО в формате JSON:
{"verdict": "correct" | "partially_correct" | "incorrect", "reasoning": "<краткое объяснение>"}
"""


def judge_answer(question: str, agent_answer: str, reference_answer: str) -> JudgeResult:
    """Один вызов LLM-судьи для оценки одного ответа агента."""
    user_message = (
        f"Вопрос: {question}\n\n"
        f"Ответ агента: {agent_answer}\n\n"
        f"Эталонный ответ: {reference_answer}"
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,  # судья должен быть стабильным, не творческим
        response_format={"type": "json_object"},
    )

    raw = json.loads(response.choices[0].message.content)
    return JudgeResult(**raw)


def judge_batch(items: list[dict]) -> dict:
    """
    Оценивает несколько ответов и считает распределение вердиктов.

    items: список {"question": str, "agent_answer": str, "reference_answer": str}
    """
    results = []

    for item in items:
        verdict = judge_answer(
            item["question"],
            item["agent_answer"],
            item["reference_answer"],
        )
        results.append({**item, "verdict": verdict.verdict.value, "reasoning": verdict.reasoning})
        print(f"[{verdict.verdict.value}] {item['question'][:60]}...")

    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    partial = sum(1 for r in results if r["verdict"] == "partially_correct")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")

    return {
        "results": results,
        "summary": {
            "total": total,
            "correct": correct,
            "partially_correct": partial,
            "incorrect": incorrect,
            "correct_rate": round(correct / total, 3) if total else 0.0,
        },
    }
