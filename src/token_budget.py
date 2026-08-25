"""
Token budget и sliding window — управление длиной контекста в диалоге.

Проблема: у каждой модели есть лимит контекста (context window).
Длинный диалог рано или поздно упрётся в этот лимит.
Sliding window — простое решение: храним только последние N сообщений,
остальное "выпадает" из активной памяти.

Установка:
    pip install tiktoken
"""

import tiktoken

# кодировка для gpt-4o-mini и большинства современных моделей OpenAI
ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Считает количество токенов в тексте — примерно, но достаточно точно для budget-планирования."""
    return len(ENCODING.encode(text))


def count_messages_tokens(messages: list[dict]) -> int:
    """
    Считает суммарное число токенов во всех messages диалога.

    +4 токена на сообщение — грубая оценка накладных расходов
    на служебные токены роли (role: system/user/assistant).
    Для точного подсчёта в проде используй tiktoken рекомендации OpenAI.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += count_tokens(content) + 4
    return total


def apply_sliding_window(
    messages: list[dict],
    max_tokens: int = 4000,
    keep_system: bool = True,
) -> list[dict]:
    """
    Обрезает историю messages так, чтобы уложиться в max_tokens.

    Стратегия: всегда сохраняем system message (если есть) и
    самые СВЕЖИЕ сообщения — обрезаем с начала истории, а не с конца.
    Свежий контекст обычно важнее для ответа на текущий вопрос.
    """
    if not messages:
        return messages

    system_msg = None
    rest_messages = messages

    if keep_system and messages[0].get("role") == "system":
        system_msg = messages[0]
        rest_messages = messages[1:]

    system_tokens = count_tokens(system_msg["content"]) + 4 if system_msg else 0
    budget = max_tokens - system_tokens

    # идём с конца (самые свежие сообщения) и набираем, пока хватает бюджета
    kept_reversed = []
    used_tokens = 0

    for msg in reversed(rest_messages):
        msg_tokens = count_tokens(msg.get("content", "")) + 4
        if used_tokens + msg_tokens > budget:
            break
        kept_reversed.append(msg)
        used_tokens += msg_tokens

    kept = list(reversed(kept_reversed))

    result = [system_msg] + kept if system_msg else kept
    return result


if __name__ == "__main__":
    sample_messages = [
        {"role": "system", "content": "Ты — полезный ассистент."},
        {"role": "user", "content": "Привет, расскажи про Python."},
        {"role": "assistant", "content": "Python — это язык программирования." * 50},
        {"role": "user", "content": "А что насчёт FastAPI?"},
        {"role": "assistant", "content": "FastAPI — это веб-фреймворк." * 50},
        {"role": "user", "content": "Спасибо, последний вопрос: что такое Docker?"},
    ]

    total = count_messages_tokens(sample_messages)
    print(f"Всего токенов в истории: {total}")

    trimmed = apply_sliding_window(sample_messages, max_tokens=100)
    print(f"\nПосле sliding window (budget=100): {len(trimmed)} сообщений из {len(sample_messages)}")
    for m in trimmed:
        print(f"  [{m['role']}] {m['content'][:50]}...")