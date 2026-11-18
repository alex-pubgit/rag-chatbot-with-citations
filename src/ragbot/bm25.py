"""A compact Okapi BM25 keyword index."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from ragbot.text import terms


class BM25Index:
    """BM25 over a fixed list of documents, addressed by their position in that list."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._size = len(documents)
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for row, document in enumerate(documents):
            counts = Counter(terms(document))
            self._lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self._postings.setdefault(term, []).append((row, frequency))
        self._average_length = sum(self._lengths) / self._size if self._size else 0.0

    def idf(self, term: str) -> float:
        """Inverse document frequency of a stemmed term. Unseen terms get the highest value."""
        matches = len(self._postings.get(term, ()))
        return math.log(1.0 + (self._size - matches + 0.5) / (matches + 0.5))

    def search(self, query: str, top_n: int) -> list[tuple[int, float]]:
        """Return up to ``top_n`` (row, score) pairs with a positive score, best first."""
        scores: dict[int, float] = {}
        for term in dict.fromkeys(terms(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for row, frequency in postings:
                length_norm = 1.0 - self._b + self._b * self._lengths[row] / self._average_length
                gain = frequency * (self._k1 + 1.0) / (frequency + self._k1 * length_norm)
                scores[row] = scores.get(row, 0.0) + idf * gain
        ranked = sorted(scores.items(), key=lambda pair: (-round(pair[1], 9), pair[0]))
        return ranked[: max(top_n, 0)]
