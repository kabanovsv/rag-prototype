"""Прогоняет контрольные вопросы и пишет отчёт в eval_results.md.

Запуск: rag/.venv/bin/python rag/eval.py

Для каждого вопроса проверяем две вещи отдельно:
1) попал ли нужный источник в топ-6 найденных чанков (качество поиска);
2) содержит ли ответ ключевые факты (качество генерации).
Так сразу видно, где именно сломалось: в поиске или в модели.
"""
import datetime
import os
import re
import time

from ask import MIN_SCORE, REFUSAL, build_prompt, load_chunks
from ollama_client import LLM_MODEL, generate
from retrieval import HybridIndex

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "eval_results.md")

# must — основы слов, которые обязаны быть в правильном ответе.
CASES = [
    {
        "question": "Что такое BPMN и для чего используется?",
        "source": "bpmn",
        "must": ["bpmn", "моделир", "бизнес-процесс"],
        "reference": "Нотация и модель для моделирования бизнес-процессов.",
    },
    {
        "question": "Какие три типа объектов потока управления есть в BPMN?",
        "source": "bpmn",
        "must": ["событ", "действ", "логическ"],
        "reference": "События, действия и логические операторы (шлюзы).",
    },
    {
        "question": "Из каких частей состоит JWT?",
        "source": "jwt",
        "must": ["заголов", "payload", "подпис"],
        "reference": "Заголовок, полезная нагрузка и подпись.",
    },
    {
        "question": "Как Kafka хранит сообщения?",
        "source": "kafka",
        "must": ["раздел", "смещен", "сообщени"],
        "reference": "Сообщения в темах, разбитых на разделы; внутри раздела упорядочены по смещениям (offset).",
    },
    {
        "question": "Что такое под (pod) в Kubernetes?",
        "source": "kubernetes",
        "must": ["контейнер", "узл"],
        "reference": "Базовая единица запуска: один или несколько контейнеров на одном узле.",
    },
    {
        "question": "Что такое микросервисная архитектура?",
        "source": "microservices",
        "must": ["модул", "связан"],
        "reference": "Архитектура из небольших, слабо связанных и независимо развёртываемых модулей.",
    },
    {
        "question": "Что описывает OpenAPI?",
        "source": "openapi",
        "must": ["rest", "спецификац"],
        "reference": "Формализованная спецификация для описания REST-сервисов.",
    },
    {
        "question": "Что изучает системный анализ?",
        "source": "system-analysis",
        "must": ["систем", "проблем"],
        "reference": "Прикладное направление теории систем для сложных слабоформализуемых проблем.",
    },
    {
        "question": "Что такое UML?",
        "source": "uml",
        "must": ["язык", "моделир"],
        "reference": "Унифицированный язык графического моделирования.",
    },
    {
        "question": "Какие возможности есть у PostgreSQL?",
        "source": "postgresql",
        "must": ["репликац", "explain"],
        "reference": "Среди прочего: репликация, профилирование запросов через EXPLAIN.",
    },
]

# Вопросы, ответов на которые в базе нет. Правильное поведение — честный отказ.
TRAPS = [
    "Что такое SLA?",
    "Как приготовить борщ?",
    "Какая погода в Москве?",
]


def keyword_hits(answer, stems):
    text = answer.lower()
    return [stem for stem in stems if re.search(r"\b" + re.escape(stem), text)]


def run_case(case, index):
    started = time.time()
    found = index.search(case["question"])
    best_score = found[0]["vector"]
    source_hit = any(item["chunk"]["source"] == case["source"] for item in found)

    if best_score < MIN_SCORE:
        answer = REFUSAL
        used_model = False
    else:
        answer = generate(build_prompt(case["question"], found))
        used_model = True

    hits = keyword_hits(answer, case["must"])
    return {
        **case,
        "answer": answer,
        "source_hit": source_hit,
        "hits": hits,
        "correct": source_hit and len(hits) == len(case["must"]),
        "best_score": best_score,
        "used_model": used_model,
        "seconds": time.time() - started,
    }


def run_trap(question, index):
    started = time.time()
    found = index.search(question)
    best_score = found[0]["vector"]
    if best_score < MIN_SCORE:
        answer = REFUSAL
    else:
        answer = generate(build_prompt(question, found))
    return {
        "question": question,
        "answer": answer,
        "refused": REFUSAL in answer,
        "best_score": best_score,
        "seconds": time.time() - started,
    }


def write_report(cases, traps):
    correct = sum(1 for case in cases if case["correct"])
    retrieved = sum(1 for case in cases if case["source_hit"])
    refused = sum(1 for trap in traps if trap["refused"])

    lines = [
        "# Контрольные вопросы: результаты",
        "",
        f"Дата: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}",
        f"Модель: `{LLM_MODEL}`, эмбеддинги: `nomic-embed-text`, "
        f"гибридный поиск (вектор + BM25), top-6 фрагментов.",
        "",
        "## Сводка",
        "",
        f"- Правильных ответов: **{correct} из {len(cases)}**.",
        f"- Нужный источник попал в топ-6: **{retrieved} из {len(cases)}**.",
        f"- Честных отказов на вопросах вне базы: **{refused} из {len(traps)}**.",
        "",
        "## Вопросы",
        "",
    ]

    for number, case in enumerate(cases, start=1):
        mark = "✅" if case["correct"] else "❌"
        lines += [
            f"### {number}. {case['question']} {mark}",
            "",
            f"- Источник в топ-6: {'да' if case['source_hit'] else '**нет**'} "
            f"(лучшее сходство {case['best_score']:.2f}).",
            f"- Ключевые факты: {len(case['hits'])} из {len(case['must'])} "
            f"({', '.join(case['hits']) or 'ни одного'}).",
            f"- Эталон: {case['reference']}",
            f"- Время: {case['seconds']:.1f} с.",
            "",
            "**Ответ:**",
            "",
            case["answer"],
            "",
        ]

    lines += ["## Вопросы вне базы (ловушки)", ""]
    for number, trap in enumerate(traps, start=1):
        mark = "✅" if trap["refused"] else "❌"
        lines += [
            f"### {number}. {trap['question']} {mark}",
            "",
            f"- Лучшее сходство: {trap['best_score']:.2f}.",
            f"- Ответ: {trap['answer']}",
            f"- Время: {trap['seconds']:.1f} с.",
            "",
        ]

    with open(RESULT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return correct, retrieved, refused


def main():
    index = HybridIndex(load_chunks())
    cases = []
    for number, case in enumerate(CASES, start=1):
        result = run_case(case, index)
        cases.append(result)
        print(f"[{number}/{len(CASES)}] {'OK ' if result['correct'] else 'FAIL'} "
              f"{case['question']} ({result['seconds']:.1f} с)")

    traps = []
    for number, question in enumerate(TRAPS, start=1):
        result = run_trap(question, index)
        traps.append(result)
        print(f"[ловушка {number}/{len(TRAPS)}] "
              f"{'OK ' if result['refused'] else 'FAIL'} {question}")

    correct, retrieved, refused = write_report(cases, traps)
    print(f"\nПравильных ответов: {correct} из {len(CASES)}")
    print(f"Нужный источник в топ-6: {retrieved} из {len(CASES)}")
    print(f"Честных отказов: {refused} из {len(TRAPS)}")
    print(f"Отчёт: {RESULT_PATH}")


if __name__ == "__main__":
    main()
