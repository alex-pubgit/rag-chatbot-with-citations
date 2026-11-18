"""Offline default embedder: deterministic feature hashing, no model and no download.

This is a lexical stand-in, not a semantic model. Two texts are similar when
they share words, word stems or character n-grams (which also forgives small
spelling differences). It does not know that "cost" and "fee" mean the same
thing. Use ``fastembed`` or an OpenAI-compatible endpoint for that.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from ragbot.text import STOPWORDS, stem, tokenize

_NGRAM_SIZES = (3, 4, 5)


def _slot(feature: str, dimension: int) -> tuple[int, float]:
    """Map a feature to a vector position and a sign, the same on every machine and run."""
    value = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
    return (value >> 1) % dimension, 1.0 if value & 1 else -1.0


@lru_cache(maxsize=100_000)
def _token_features(token: str, dimension: int) -> tuple[tuple[int, float], ...]:
    """Positions and weights for one token: its stem, plus its character n-grams.

    The n-grams of a token share a total weight equal to the stem's, so an exact
    word match counts twice as much as a near match such as "bridge" / "bridging".
    """
    features = [_slot(f"w:{stem(token)}", dimension)]
    if token.isalpha() and len(token) >= 3:
        padded = f"<{token}>"
        grams = [padded[i : i + n] for n in _NGRAM_SIZES for i in range(len(padded) - n + 1)]
        share = 1.0 / math.sqrt(len(grams))
        for gram in grams:
            index, sign = _slot(f"g:{gram}", dimension)
            features.append((index, sign * share))
    return tuple(features)


class HashingEmbedder:
    """Hash word stems and character n-grams into a fixed-size unit vector."""

    # Calibrated on the sample golden set with `ragbot eval --calibrate`.
    # The README's Quality section explains the rule and how to recalibrate.
    default_min_score: float | None = 0.21

    def __init__(self, dimension: int = 2048) -> None:
        self.dimension = dimension
        self.fingerprint = f"hashing-v1-{dimension}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self.embed_query(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        counts = Counter(token for token in tokenize(text) if token not in STOPWORDS)
        for token, count in counts.items():
            weight = 1.0 + math.log(count)
            for index, value in _token_features(token, self.dimension):
                vector[index] += weight * value
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector
