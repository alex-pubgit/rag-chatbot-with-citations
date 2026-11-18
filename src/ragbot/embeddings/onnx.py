"""A real embedding model that runs locally through ONNX (fastembed). No API key."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ragbot.embeddings import unit_rows
from ragbot.errors import ConfigError

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Refusal thresholds calibrated on the sample golden set with `ragbot eval --calibrate`.
_CALIBRATED_MIN_SCORE = {DEFAULT_MODEL: 0.54}


class FastEmbedEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ConfigError(
                "RAGBOT_EMBEDDER=fastembed needs an extra package. "
                'Install it with: pip install "ragbot[fastembed]"'
            ) from exc
        self._model = TextEmbedding(model_name=model)
        self.fingerprint = f"fastembed:{model}"
        self.default_min_score = _CALIBRATED_MIN_SCORE.get(model)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return unit_rows(np.vstack(list(self._model.passage_embed(list(texts)))))

    def embed_query(self, text: str) -> np.ndarray:
        return unit_rows(np.vstack(list(self._model.query_embed(text))))[0]
