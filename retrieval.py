"""Гибридный поиск по чанкам: векторное сходство + BM25 по словам.

Векторный поиск ловит смысл, но промахивается на точных терминах и
вопросах-перечислениях. BM25 (классический поиск по словам) ловит совпадения
терминов, но не понимает синонимы. Смешиваем оба результата 50 на 50.
"""
import math
import re
from collections import Counter

import numpy as np

from ollama_client import embed_query

TOP_K = 6
VECTOR_WEIGHT = 0.5
STEM_LENGTH = 5  # грубая основа слова: «моделирование» -> «модел»


def tokenize(text):
    """Слова длиннее двух символов, обрезанные до грубой основы."""
    words = re.findall(r"[а-яёa-z0-9]+", text.lower())
    return [word[:STEM_LENGTH] for word in words if len(word) > 2]


def _normalize(values):
    """Приводит оценки к диапазону 0..1, чтобы можно было складывать."""
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


class Bm25:
    """BM25: слово весит тем больше, чем оно реже в базе и чаще в чанке."""

    def __init__(self, texts):
        self.documents = [tokenize(text) for text in texts]
        self.count = len(self.documents)
        self.average_length = sum(len(doc) for doc in self.documents) / self.count
        self.term_frequencies = [Counter(doc) for doc in self.documents]

        document_frequency = Counter()
        for doc in self.documents:
            document_frequency.update(set(doc))
        self.document_frequency = document_frequency

    def scores(self, query):
        query_terms = tokenize(query)
        result = np.zeros(self.count, dtype=np.float32)

        for index, frequencies in enumerate(self.term_frequencies):
            length = len(self.documents[index])
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (self.count - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (0.25 + 0.75 * length / self.average_length)
                result[index] += idf * frequency * 2.5 / denominator

        return result


class HybridIndex:
    """Держит векторы и BM25-статистику, ищет top-K по смешанной оценке."""

    def __init__(self, chunks):
        self.chunks = chunks
        matrix = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix
        self.bm25 = Bm25([chunk["text"] for chunk in chunks])

    def search(self, question, top_k=TOP_K):
        vector = np.array(embed_query(question), dtype=np.float32)
        vector /= np.linalg.norm(vector)
        vector_scores = self.matrix @ vector
        lexical_scores = self.bm25.scores(question)

        # Обе оценки приводим к 0..1, иначе BM25 (десятки) перевесит косинус.
        vector_norm = _normalize(vector_scores)
        lexical_norm = _normalize(lexical_scores)
        combined = VECTOR_WEIGHT * vector_norm + (1 - VECTOR_WEIGHT) * lexical_norm

        order = np.argsort(-combined)[:top_k]
        return [
            {
                "score": float(combined[index]),      # итоговая смешанная оценка
                "vector": float(vector_scores[index]),  # косинусное сходство
                "lexical": float(lexical_norm[index]),  # BM25, приведённый к 0..1
                "chunk": self.chunks[index],
            }
            for index in order
        ]
