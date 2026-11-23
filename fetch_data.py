"""Скачивает статьи Википедии для учебной базы знаний.

Запуск: rag/.venv/bin/python rag/fetch_data.py
Файлы кладутся в rag/data/*.txt. Повторный запуск перезаписывает их.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Тематика: интеграции, данные, архитектура, анализ.
ARTICLES = {
    "REST": "rest",
    "JSON Web Token": "jwt",
    "Apache Kafka": "kafka",
    "PostgreSQL": "postgresql",
    "Микросервисная архитектура": "microservices",
    "Kubernetes": "kubernetes",
    "BPMN": "bpmn",
    "Системный анализ": "system-analysis",
    "OpenAPI Specification": "openapi",
    "UML": "uml",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
API_URL = "https://ru.wikipedia.org/w/api.php"
USER_AGENT = "RAG-study-prototype/1.0 (educational use)"

# Служебные разделы в конце статьи не нужны для ответов по существу.
CUT_MARKERS = [
    "\n== Ссылки ==",
    "\n== Примечания ==",
    "\n== Литература ==",
    "\n== Источники ==",
    "\n== См. также ==",
]


def fetch_article(title: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    # Википедия ограничивает частоту запросов: на 429 ждём и пробуем снова.
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code != 429:
                raise
            time.sleep(5 * (attempt + 1))
    else:
        raise last_error

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        raise LookupError(f"статья не найдена: {title}")
    text = page.get("extract", "").strip()
    if not text:
        raise LookupError(f"пустой текст: {title}")
    return text


def clean(text: str) -> str:
    for marker in CUT_MARKERS:
        index = text.find(marker)
        if index != -1:
            text = text[:index]
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    force = "--force" in sys.argv
    total = 0
    for title, slug in ARTICLES.items():
        path = os.path.join(DATA_DIR, slug + ".txt")
        if os.path.exists(path) and not force:
            print(f"ЕСТЬ     {title} -> {slug}.txt")
            continue
        try:
            text = clean(fetch_article(title))
        except Exception as error:  # noqa: BLE001 — в учебном скрипте достаточно предупреждения
            print(f"ПРОПУСК  {title}: {error}")
            continue
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
        total += 1
        print(f"OK       {title} -> {slug}.txt, {len(text)} символов")
        time.sleep(1)  # вежливая пауза между запросами к API
    print(f"\nСкачано статей за этот запуск: {total}")


if __name__ == "__main__":
    main()
