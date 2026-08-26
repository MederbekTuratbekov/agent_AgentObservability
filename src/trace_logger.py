"""
Trace logging — записываем каждый шаг работы агента для отладки.

Зачем: когда агент делает несколько шагов подряд (reasoning, tool calls),
без логов невозможно понять ГДЕ он ошибся — на этапе рассуждения,
выбора инструмента, или интерпретации результата.

Здесь — упрощённая обёртка в стиле Langfuse/LangSmith: каждый вызов LLM
или tool записывается со временем выполнения и сохраняется в JSON.
"""

import json
import time
from datetime import datetime
from contextlib import contextmanager


class TraceLogger:
    """
    Собирает трейс одного прогона агента: последовательность
    событий (LLM-вызовы, tool-вызовы) с временными метками.
    """

    def __init__(self, trace_name: str):
        self.trace_name = trace_name
        self.trace_id = f"{trace_name}_{int(time.time())}"
        self.events: list[dict] = []
        self.start_time = time.time()

    def log_event(self, event_type: str, name: str, input_data, output_data, duration_ms: float):
        """
        Записывает одно событие в трейс.

        event_type: "llm_call" | "tool_call"
        name: название конкретного вызова (модель или инструмент)
        """
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "name": name,
            "input": str(input_data)[:500],
            "output": str(output_data)[:500],
            "duration_ms": round(duration_ms, 1),
        })

    @contextmanager
    def span(self, event_type: str, name: str, input_data=None):
        """
        Context manager для замера времени выполнения блока кода.

        Использование:
            with tracer.span("tool_call", "calculator", input_data="2+2") as record:
                result = calculator(...)
                record(result)
        """
        start = time.time()
        output_holder = {"value": None}

        def record(output_data):
            output_holder["value"] = output_data

        yield record

        duration_ms = (time.time() - start) * 1000
        self.log_event(event_type, name, input_data, output_holder["value"], duration_ms)

    def total_duration_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    def save(self, path: str = None):
        """Сохраняет трейс в JSON-файл для последующего анализа."""
        path = path or f"{self.trace_id}.json"
        report = {
            "trace_id": self.trace_id,
            "trace_name": self.trace_name,
            "total_duration_ms": round(self.total_duration_ms(), 1),
            "num_events": len(self.events),
            "events": self.events,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Трейс сохранён: {path}")
        return path

    def print_summary(self):
        """Печатает краткую сводку трейса в консоль."""
        print(f"\n=== Trace: {self.trace_name} ===")
        print(f"Всего событий: {len(self.events)}")
        print(f"Общее время: {self.total_duration_ms():.1f} ms\n")

        for e in self.events:
            print(f"  [{e['event_type']}] {e['name']} — {e['duration_ms']}ms")
