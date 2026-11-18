from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from ragbot.documents import Chunk
from ragbot.embeddings.hashing import HashingEmbedder
from ragbot.errors import IndexStoreError
from ragbot.index import Index
from ragbot.retrieval import CANDIDATES, Retriever
from ragbot.vector_store import NumpyVectorStore

TEXTS = [
    ("swap.md", "Swap fees", "The swap fee is 0.25% on every network."),
    ("bridge.md", "Bridging", "The bridge fee is 0.05% with a minimum of 1 USDT equivalent."),
    ("staking.md", "Staking", "Unstaking takes 7 days. Rewards are distributed every Monday."),
    ("gas.md", "Network fees", "Gas is paid in the native coin and goes to the network."),
    ("api.md", "Public API", "The public API allows 60 requests per minute per IP."),
]


def _chunks() -> list[Chunk]:
    return [Chunk(f"{source}#0", source, title, (), 0, text) for source, title, text in TEXTS]


def _retriever(mode: str = "hybrid", reranker: object | None = None) -> Retriever:
    embedder = HashingEmbedder(dimension=512)
    chunks = _chunks()
    vectors = NumpyVectorStore(embedder.embed_documents([c.index_text for c in chunks]))
    return Retriever(chunks, vectors, embedder, mode=mode, reranker=reranker)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "dense"])
def test_every_mode_puts_the_matching_passage_first(mode: str) -> None:
    passages = _retriever(mode).retrieve("What is the bridge fee?", top_k=3)

    assert passages[0].chunk.source == "bridge.md"
    assert len(passages) <= 3


def test_score_is_the_cosine_between_question_and_passage() -> None:
    embedder = HashingEmbedder(dimension=512)
    question = "How many API requests per minute?"

    passages = _retriever("hybrid").retrieve(question, top_k=5)

    for passage in passages:
        expected = float(
            embedder.embed_query(question) @ embedder.embed_query(passage.chunk.index_text)
        )
        assert passage.score == pytest.approx(max(expected, 0.0), abs=1e-5)
        assert 0.0 <= passage.score <= 1.0
    assert passages[0].chunk.source == "api.md"
    assert passages[0].score > 0.3


def test_bm25_mode_only_returns_passages_that_share_a_term() -> None:
    passages = _retriever("bm25").retrieve("unstaking period", top_k=5)

    assert [p.chunk.source for p in passages] == ["staking.md"]


def test_dense_mode_fills_top_k_even_without_shared_terms() -> None:
    passages = _retriever("dense").retrieve("unstaking period", top_k=4)

    assert len(passages) == 4
    assert passages[0].chunk.source == "staking.md"


def test_question_without_any_match_gets_low_scores() -> None:
    passages = _retriever("hybrid").retrieve("How do I bake sourdough bread?", top_k=3)

    assert all(passage.score < 0.1 for passage in passages)


def test_reranker_reorders_the_fused_candidates() -> None:
    seen: list[list[str]] = []

    class ReverseReranker:
        def rerank(self, question: str, chunks: Sequence[Chunk]) -> list[int]:
            seen.append([chunk.source for chunk in chunks])
            return list(reversed(range(len(chunks))))

    plain = _retriever("hybrid").retrieve("network fee", top_k=len(TEXTS))
    reranked = _retriever("hybrid", ReverseReranker()).retrieve("network fee", top_k=2)

    assert len(seen[0]) == len(plain) > 2, "the reranker must see more than top_k candidates"
    assert len(TEXTS) <= CANDIDATES
    assert [p.chunk.source for p in reranked] == [p.chunk.source for p in reversed(plain)][:2]


def test_from_index_rejects_an_index_built_with_another_embedder() -> None:
    embedder = HashingEmbedder(dimension=512)
    chunks = _chunks()
    index = Index(
        embedder="some-other-model",
        chunk_size=160,
        chunk_overlap=30,
        files={},
        chunks=chunks,
        embeddings=np.zeros((len(chunks), 512), dtype=np.float32),
    )

    with pytest.raises(IndexStoreError, match="built with the embedder 'some-other-model'"):
        Retriever.from_index(index, embedder)


def test_vector_store_orders_by_similarity_then_row() -> None:
    store = NumpyVectorStore(np.array([[1, 0], [0, 1], [1, 0], [0.6, 0.8]], dtype=np.float32))
    query = np.array([1, 0], dtype=np.float32)

    assert store.search(query, top_n=3) == [(0, 1.0), (2, 1.0), (3, pytest.approx(0.6))]
    assert store.similarity(query, [3, 1]) == [pytest.approx(0.6), 0.0]
    assert store.search(query, top_n=0) == []
    assert store.similarity(query, []) == []
