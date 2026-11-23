"""Собирает индекс: режет статьи на чанки и считает эмбеддинги.

Запуск: rag/.venv/bin/python rag/build_index.py
Результат: rag/index.json — чанки с текстом, источником и вектором.
"""
import json
import os
import re
import time

from fetch_data import ARTICLES
from ollama_client import EMBED_MODEL, embed_documents

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(BASE_DIR, "index.json")

CHUNK_SIZE = 800  # максимальная длина чанка в символах
OVERLAP = 150     # сколько символов предыдущего чанка повторяем в следующем
EMBED_BATCH = 16  # сколько чанков отправляем в Ollama за один запрос

# Тело чанка без перекрытия: оставляем место под 150 символов хвоста.
MAX_UNIT = CHUNK_SIZE - OVERLAP - 2

# slug файла -> человеческое название статьи (для ссылок в ответе)
TITLES = {slug: title for title, slug in ARTICLES.items()}


def split_into_units(text, max_unit=MAX_UNIT):
    """Делит текст на абзацы; слишком длинный абзац режет на предложения."""
    units = []
    for paragraph in re.split(r"\n\s*\n", text):
        # Внутри абзаца переносы строк — просто пробелы.
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= max_unit:
            units.append(paragraph)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            if len(sentence) <= max_unit:
                units.append(sentence)
            else:
                # Совсем длинное предложение (например, список) режем жёстко.
                for start in range(0, len(sentence), max_unit):
                    units.append(sentence[start:start + max_unit])
    return units


def _joined_length(parts):
    """Длина текста, склеенного через пустую строку."""
    return sum(len(part) for part in parts) + 2 * max(len(parts) - 1, 0)


def make_chunks(units, size=CHUNK_SIZE, overlap=OVERLAP):
    """Собирает абзацы в чанки не больше size, с перекрытием хвоста.

    Сначала копим «тела» чанков с запасом под перекрытие, потом приклеиваем
    к каждому телу хвост предыдущего — ровно overlap символов (с точностью
    до границы слова).
    """
    body_limit = size - overlap - 2

    bodies = []
    current = []
    for unit in units:
        if current and _joined_length(current) + 2 + len(unit) > body_limit:
            bodies.append("\n\n".join(current))
            current = []
        current.append(unit)
    if current:
        bodies.append("\n\n".join(current))

    chunks = []
    for number, body in enumerate(bodies):
        if number == 0 or overlap <= 0:
            chunks.append(body)
            continue
        tail = bodies[number - 1][-overlap:]
        # Не начинаем перекрытие с середины слова.
        space = tail.find(" ")
        if space != -1:
            tail = tail[space + 1:]
        chunks.append(tail + "\n\n" + body)
    return chunks


def main():
    started = time.time()
    chunks = []

    for slug in sorted(TITLES):
        path = os.path.join(DATA_DIR, slug + ".txt")
        if not os.path.exists(path):
            print(f"ПРОПУСК  {slug}: нет файла {path}")
            continue
        with open(path, encoding="utf-8") as file:
            text = file.read()

        article_chunks = make_chunks(split_into_units(text))
        for number, chunk_text in enumerate(article_chunks, start=1):
            chunks.append({
                "id": f"{slug}-{number:03d}",
                "source": slug,
                "title": TITLES[slug],
                "number": number,
                "text": chunk_text,
            })
        print(f"{TITLES[slug]:35} чанков: {len(article_chunks)}")

    if not chunks:
        raise SystemExit("Нет данных: сначала запустите fetch_data.py")

    print(f"\nВсего чанков: {len(chunks)}. Считаю эмбеддинги...")
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start:start + EMBED_BATCH]
        vectors = embed_documents([chunk["text"] for chunk in batch])
        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = vector
        done = min(start + EMBED_BATCH, len(chunks))
        print(f"  {done} из {len(chunks)}")

    index = {
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    with open(INDEX_PATH, "w", encoding="utf-8") as file:
        json.dump(index, file, ensure_ascii=False)

    sizes = [len(chunk["text"]) for chunk in chunks]
    print(f"\nГотово за {time.time() - started:.1f} с")
    print(f"Чанков: {len(chunks)} | символов в чанке: {min(sizes)}–{max(sizes)}, "
          f"в среднем {sum(sizes) // len(sizes)}")
    print(f"Размерность вектора: {len(chunks[0]['embedding'])}")
    print(f"Файл: {INDEX_PATH} ({os.path.getsize(INDEX_PATH) / 1024 / 1024:.1f} МБ)")

    example = chunks[0]
    print(f"\nПример чанка {example['id']} ({example['title']}):")
    print(example["text"][:300].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
