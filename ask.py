"""Отвечает на вопрос по собранному индексу.

Запуск: rag/.venv/bin/python rag/ask.py "ваш вопрос"
Ключ --show-prompt печатает промпт, который получает модель.
"""
import json
import os
import sys
import time

from ollama_client import LLM_MODEL, generate
from retrieval import HybridIndex

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.json")

# Грубый фильтр: ниже этого сходства даже не зовём модель.
# Абсолютное значение косинуса — не вероятность, порог подобран на нескольких
# вопросах. В реальном проекте его калибруют на размеченном наборе.
MIN_SCORE = 0.70

REFUSAL = "В предоставленных документах нет ответа на этот вопрос."

PROMPT_TEMPLATE = """Ты — ассистент, который отвечает на вопросы строго по фрагментам документов.

Правила:
- Используй только информацию из фрагментов ниже.
- Если во фрагментах нет ответа, ответь ровно так: «{refusal}»
- Не додумывай факты, цифры и названия.
- Отвечай кратко, по-русски.
- Если ссылаешься на фрагмент, указывай его номер в квадратных скобках, например [1].

Фрагменты:
{fragments}

Вопрос: {question}
Ответ:"""


def load_chunks():
    if not os.path.exists(INDEX_PATH):
        raise SystemExit(
            "Нет индекса. Сначала запустите: rag/.venv/bin/python rag/build_index.py"
        )
    with open(INDEX_PATH, encoding="utf-8") as file:
        return json.load(file)["chunks"]


def build_prompt(question, found):
    fragments = []
    for number, item in enumerate(found, start=1):
        chunk = item["chunk"]
        fragments.append(
            f"[{number}] «{chunk['title']}», фрагмент {chunk['number']}:\n{chunk['text']}"
        )
    return PROMPT_TEMPLATE.format(
        refusal=REFUSAL,
        fragments="\n\n".join(fragments),
        question=question,
    )


def main():
    words = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if not words:
        print('Использование: python rag/ask.py "ваш вопрос" [--show-prompt]')
        return
    question = " ".join(words)
    show_prompt = "--show-prompt" in sys.argv

    chunks = load_chunks()
    found = HybridIndex(chunks).search(question)

    print(f"Вопрос: {question}\n")
    print("Найдено в индексе:")
    for number, item in enumerate(found, start=1):
        chunk = item["chunk"]
        print(f"  [{number}] итог {item['score']:.2f} — {chunk['title']}, "
              f"фрагмент {chunk['number']} (вектор {item['vector']:.2f}, "
              f"слова {item['lexical']:.2f})")

    best_score = found[0]["vector"]
    if best_score < MIN_SCORE:
        print(f"\nЛучшее векторное сходство {best_score:.2f} ниже порога {MIN_SCORE:.2f}.")
        print(REFUSAL)
        return

    prompt = build_prompt(question, found)
    if show_prompt:
        print("\n--- Промпт для модели ---")
        print(prompt)
        print("--- Конец промпта ---\n")

    print(f"\nДумаю... ({LLM_MODEL})\n")
    started = time.time()
    answer = generate(prompt)

    print("Ответ:\n")
    print(answer)
    print(f"\n(сгенерировано за {time.time() - started:.1f} с)")

    if REFUSAL in answer:
        print("\nМодель не нашла ответа в базе — источники не привожу.")
        return

    print("\nИсточники:")
    for number, item in enumerate(found, start=1):
        chunk = item["chunk"]
        print(f"  [{number}] {chunk['title']}, {chunk['id']} "
              f"(вектор {item['vector']:.2f}, слова {item['lexical']:.2f})")


if __name__ == "__main__":
    main()
