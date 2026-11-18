"""Pluggable embedders. The rest of the code depends only on the ``Embedder`` protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Turns text into unit-length vectors, so a dot product is a cosine similarity."""

    fingerprint: str
    """Identifies the model. An index can only be searched with the embedder that built it."""

    default_min_score: float | None
    """Refusal threshold calibrated for this embedder on the sample golden set, if any."""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return one row per text, shape ``(len(texts), dimension)``, dtype float32."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return a single vector of shape ``(dimension,)``."""
        ...


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to length 1. All-zero rows are left as they are."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
