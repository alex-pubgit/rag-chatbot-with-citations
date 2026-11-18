"""Dense similarity search behind a small interface.

``NumpyVectorStore`` is an exact, in-memory implementation that is enough for a
few thousand passages. A vector database can replace it by implementing the
two methods of ``VectorStore``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    def search(self, query: np.ndarray, top_n: int) -> list[tuple[int, float]]:
        """Return up to ``top_n`` (row, cosine similarity) pairs, best first."""
        ...

    def similarity(self, query: np.ndarray, rows: Sequence[int]) -> list[float]:
        """Cosine similarity between ``query`` and each of the given rows."""
        ...


class NumpyVectorStore:
    """Exact cosine search over a matrix of unit-length row vectors."""

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    def search(self, query: np.ndarray, top_n: int) -> list[tuple[int, float]]:
        if top_n <= 0 or not len(self._matrix):
            return []
        scores = self._scores(self._matrix, query)
        order = np.lexsort((np.arange(len(scores)), -scores))[:top_n]
        return [(int(row), float(scores[row])) for row in order]

    def similarity(self, query: np.ndarray, rows: Sequence[int]) -> list[float]:
        if not rows:
            return []
        return [float(score) for score in self._scores(self._matrix[list(rows)], query)]

    @staticmethod
    def _scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
        # Rounding keeps the ranking identical across platforms and BLAS builds.
        return np.round(matrix @ query.astype(np.float32), 6)
