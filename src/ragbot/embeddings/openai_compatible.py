"""Embeddings from any OpenAI-compatible ``/embeddings`` endpoint."""

from __future__ import annotations

from collections.abc import Sequence

import httpx2 as httpx
import numpy as np

from ragbot.embeddings import unit_rows
from ragbot.errors import ProviderError
from ragbot.remote import open_client, post_json

BATCH_SIZE = 64


class OpenAIEmbedder:
    # Similarity scales differ from model to model, so there is no safe default
    # refusal threshold: calibrate with `ragbot eval --calibrate`.
    default_min_score: float | None = None

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._client = open_client(base_url, api_key, timeout, transport)
        self.fingerprint = f"openai:{model}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = list(texts[start : start + BATCH_SIZE])
            reply = post_json(self._client, "embeddings", {"model": self._model, "input": batch})
            try:
                # The API may return items out of order; "index" is the position in the batch.
                items = sorted(reply["data"], key=lambda item: item["index"])
                vectors = [item["embedding"] for item in items]
            except (KeyError, TypeError) as exc:
                raise ProviderError(
                    "The embeddings endpoint answered without data[].embedding. "
                    "Check that RAGBOT_EMBEDDING_BASE_URL points at an OpenAI-compatible API."
                ) from exc
            if len(vectors) != len(batch):
                raise ProviderError(
                    f"The endpoint returned {len(vectors)} vectors for {len(batch)} texts."
                )
            rows.extend(vectors)
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return unit_rows(np.asarray(rows, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
