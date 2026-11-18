from __future__ import annotations

import json

import httpx2 as httpx
import numpy as np
import pytest

from ragbot.embeddings.openai_compatible import BATCH_SIZE, OpenAIEmbedder
from ragbot.errors import ProviderError


def embedder(handler) -> OpenAIEmbedder:
    return OpenAIEmbedder(
        base_url="https://embed.test/v1", model="test-embed", transport=httpx.MockTransport(handler)
    )


def test_batches_requests_and_restores_the_order_of_the_texts():
    batches = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        batches.append(len(texts))
        # Encode each text's length in its vector, and answer in reverse order like a real API may.
        data = [{"index": i, "embedding": [float(len(text)), 1.0]} for i, text in enumerate(texts)]
        return httpx.Response(200, json={"data": data[::-1]})

    texts = ["x" * (n + 1) for n in range(BATCH_SIZE + 6)]
    vectors = embedder(handler).embed_documents(texts)

    assert batches == [BATCH_SIZE, 6]
    assert vectors.shape == (len(texts), 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert np.allclose(vectors[:, 0] / vectors[:, 1], [len(text) for text in texts], rtol=1e-4)


def test_fewer_vectors_than_texts_is_an_error():
    reply = httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
    with pytest.raises(ProviderError, match="1 vectors for 2 texts"):
        embedder(lambda _request: reply).embed_documents(["a", "b"])


def test_reply_without_embeddings_is_reported():
    reply = httpx.Response(200, json={"object": "list"})
    with pytest.raises(ProviderError, match="RAGBOT_EMBEDDING_BASE_URL"):
        embedder(lambda _request: reply).embed_query("a")
