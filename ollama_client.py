"""Единая точка общения с Ollama: эмбеддинги и генерация ответа.

Ollama — локальный сервер моделей. Запускается сам при первом запросе.
"""
import warnings

# Системный Python на macOS собран с LibreSSL, и urllib3 ругается на это.
# Для запросов к localhost (Ollama) предупреждение безопасно, поэтому убираем шум.
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import requests  # noqa: E402 — импорт после настройки предупреждений

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
# qwen3:4b на этом ноутбуке долго «рассуждает» перед ответом (минуты).
# qwen2.5:3b-instruct отвечает сразу и заметно быстрее — для прототипа важнее.
LLM_MODEL = "qwen2.5:3b-instruct"

# nomic-embed-text обучена с префиксами: у документа и у вопроса они разные.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def _embed_batch(texts):
    """Один запрос к Ollama: список текстов -> список векторов."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    response.raise_for_status()
    embeddings = response.json()["embeddings"]
    if len(embeddings) != len(texts):
        raise RuntimeError("Ollama вернула другое число векторов, чем текстов")
    return embeddings


def embed_documents(texts):
    """Векторы для чанков базы знаний (с префиксом документа)."""
    return _embed_batch([DOCUMENT_PREFIX + text for text in texts])


def embed_query(question):
    """Вектор вопроса (с префиксом запроса)."""
    return _embed_batch([QUERY_PREFIX + question])[0]


def generate(prompt):
    """Ответ модели на готовый промпт."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 400,  # страховка от бесконечного текста
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()
