"""Build the pipeline from settings: retrieve, check confidence, answer with citations."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import SecretStr

from ragbot.answer import REFUSED, Answer, Answerer, ExtractiveAnswerer
from ragbot.embeddings import Embedder
from ragbot.embeddings.hashing import HashingEmbedder
from ragbot.errors import ConfigError
from ragbot.index import Index
from ragbot.retrieval import Retriever
from ragbot.settings import Settings


@dataclass(frozen=True)
class Chatbot:
    index: Index
    retriever: Retriever
    answerer: Answerer
    min_score: float
    top_k: int

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        passages = self.retriever.retrieve(question, top_k or self.top_k)
        usable = [passage for passage in passages if passage.score >= self.min_score]
        if not usable:
            return REFUSED
        return self.answerer.answer(question, usable)


def _reveal(secret: SecretStr | None) -> str | None:
    return secret.get_secret_value() if secret else None


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hashing":
        return HashingEmbedder()
    if settings.embedder == "fastembed":
        from ragbot.embeddings.onnx import DEFAULT_MODEL, FastEmbedEmbedder

        return FastEmbedEmbedder(settings.embedding_model or DEFAULT_MODEL)
    if not settings.embedding_model:
        raise ConfigError(
            "RAGBOT_EMBEDDER=openai needs RAGBOT_EMBEDDING_MODEL: "
            "the name of the embedding model at your endpoint."
        )
    from ragbot.embeddings.openai_compatible import OpenAIEmbedder

    return OpenAIEmbedder(
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        api_key=_reveal(settings.embedding_api_key),
        timeout=settings.timeout_seconds,
    )


def _build_answerer(settings: Settings, retriever: Retriever) -> Answerer:
    if settings.answerer == "extractive":
        return ExtractiveAnswerer(retriever.bm25.idf)
    if not settings.llm_model:
        raise ConfigError(
            "RAGBOT_ANSWERER=openai needs RAGBOT_LLM_MODEL: "
            "the name of the chat model at your endpoint."
        )
    from ragbot.llm import OpenAIChatAnswerer

    return OpenAIChatAnswerer(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=_reveal(settings.llm_api_key),
        timeout=settings.timeout_seconds,
    )


def build_chatbot(
    settings: Settings, *, index: Index | None = None, embedder: Embedder | None = None
) -> Chatbot:
    embedder = embedder or build_embedder(settings)
    index = index or Index.load(settings.index_dir)
    retriever = Retriever.from_index(index, embedder, mode=settings.retrieval)
    min_score = settings.min_score if settings.min_score is not None else embedder.default_min_score
    if min_score is None:
        raise ConfigError(
            f"No refusal threshold is known for the embedder '{embedder.fingerprint}'. "
            "Run `ragbot eval --calibrate` on your golden set and set RAGBOT_MIN_SCORE."
        )
    return Chatbot(
        index=index,
        retriever=retriever,
        answerer=_build_answerer(settings, retriever),
        min_score=min_score,
        top_k=settings.top_k,
    )
