"""Hybrid retrieval: BM25 and dense search fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from ragbot.bm25 import BM25Index
from ragbot.documents import Chunk, Passage
from ragbot.embeddings import Embedder
from ragbot.fusion import reciprocal_rank_fusion
from ragbot.index import Index
from ragbot.vector_store import NumpyVectorStore, VectorStore

RetrievalMode = Literal["hybrid", "bm25", "dense"]

# How many results each retriever contributes before fusion, and how many
# fused candidates a reranker gets to reorder.
CANDIDATES = 20


class Reranker(Protocol):
    def rerank(self, question: str, chunks: Sequence[Chunk]) -> list[int]:
        """Return the positions of ``chunks`` ordered from most to least relevant."""
        ...


class Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        vectors: VectorStore,
        embedder: Embedder,
        *,
        mode: RetrievalMode = "hybrid",
        reranker: Reranker | None = None,
    ) -> None:
        self._chunks = list(chunks)
        self._vectors = vectors
        self._embedder = embedder
        self._mode = mode
        self._reranker = reranker
        self.bm25 = BM25Index([chunk.index_text for chunk in self._chunks])

    @classmethod
    def from_index(
        cls,
        index: Index,
        embedder: Embedder,
        *,
        mode: RetrievalMode = "hybrid",
        reranker: Reranker | None = None,
    ) -> Retriever:
        index.require_embedder(embedder.fingerprint)
        vectors = NumpyVectorStore(index.embeddings)
        return cls(index.chunks, vectors, embedder, mode=mode, reranker=reranker)

    def retrieve(self, question: str, top_k: int) -> list[Passage]:
        """Return the ``top_k`` best passages for the question, best first.

        Each passage carries the cosine similarity between the question and the
        passage under the configured embedder. That score is what the refusal
        threshold is compared against, whichever retrieval mode ranked it.
        """
        query = self._embedder.embed_query(question)
        depth = max(CANDIDATES, top_k)
        rankings: list[list[int]] = []
        if self._mode in ("hybrid", "bm25"):
            rankings.append([row for row, _ in self.bm25.search(question, depth)])
        if self._mode in ("hybrid", "dense"):
            rankings.append([row for row, _ in self._vectors.search(query, depth)])
        rows = [row for row, _ in reciprocal_rank_fusion(rankings)]

        if self._reranker and rows:
            pool = rows[:depth]
            order = self._reranker.rerank(question, [self._chunks[row] for row in pool])
            rows = [pool[position] for position in order]
        rows = rows[:top_k]
        scores = self._vectors.similarity(query, rows)
        return [
            Passage(chunk=self._chunks[row], score=min(1.0, max(0.0, score)))
            for row, score in zip(rows, scores, strict=True)
        ]
