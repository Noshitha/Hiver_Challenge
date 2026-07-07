from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


class SimpleRetriever:
    """
    Lightweight TF-IDF retrieval over training examples.
    No external dependencies - uses only Python stdlib for 100-minute time constraint.
    """

    def __init__(self, train_path: Path) -> None:
        self.examples: list[dict] = []
        self.idf_scores: dict[str, float] = {}
        self._load_and_index(train_path)

    def retrieve(self, query_text: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve top-k most similar training examples using TF-IDF cosine similarity.
        """
        query_vec = self._vectorize(query_text)
        scores: list[tuple[float, dict]] = []

        for example in self.examples:
            example_vec = self._vectorize(example["incoming_email"])
            similarity = self._cosine_similarity(query_vec, example_vec)
            scores.append((similarity, example))

        scores.sort(reverse=True, key=lambda x: x[0])
        return [item[1] for item in scores[:top_k]]

    def _load_and_index(self, train_path: Path) -> None:
        """Load training data and compute IDF scores."""
        with train_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

        doc_freq: Counter[str] = Counter()
        for example in self.examples:
            tokens = set(self._tokenize(example["incoming_email"]))
            doc_freq.update(tokens)

        num_docs = len(self.examples)
        self.idf_scores = {
            token: math.log((num_docs + 1) / (freq + 1)) for token, freq in doc_freq.items()
        }

    def _vectorize(self, text: str) -> dict[str, float]:
        """Convert text to TF-IDF vector."""
        tokens = self._tokenize(text)
        tf_counts = Counter(tokens)
        total_tokens = len(tokens)

        vec: dict[str, float] = {}
        for token, count in tf_counts.items():
            tf = count / total_tokens if total_tokens > 0 else 0.0
            idf = self.idf_scores.get(token, 0.0)
            vec[token] = tf * idf
        return vec

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    @staticmethod
    def _cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
        """Compute cosine similarity between two TF-IDF vectors."""
        if not vec1 or not vec2:
            return 0.0

        common_tokens = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[token] * vec2[token] for token in common_tokens)

        mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v**2 for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)
