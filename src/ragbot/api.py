"""HTTP API and the single-page web UI."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from importlib import resources

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from ragbot.chatbot import build_chatbot
from ragbot.errors import ProviderError, RagbotError
from ragbot.settings import Settings

MAX_BODY_BYTES = 16 * 1024


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20)


class CitationOut(BaseModel):
    n: int
    source: str
    title: str
    heading: str
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[CitationOut]
    latency_ms: int


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    chatbot = build_chatbot(settings)
    page = resources.files("ragbot").joinpath("static/index.html").read_text(encoding="utf-8")
    app = FastAPI(title="ragbot", version="0.1.0")

    @app.middleware("http")
    async def limit_body_size(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        declared = request.headers.get("content-length", "0")
        if not declared.isdigit() or int(declared) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request body too large."}, status_code=413)
        return await call_next(request)

    @app.exception_handler(RagbotError)
    async def report_error(_request: Request, exc: RagbotError) -> JSONResponse:
        # A failing upstream model is a bad gateway; anything else is a server-side fault.
        status = 502 if isinstance(exc, ProviderError) else 500
        return JSONResponse({"detail": str(exc)}, status_code=status)

    @app.get("/health")
    def health() -> dict[str, str | int]:
        return {
            "status": "ok",
            "documents": len(chatbot.index.files),
            "passages": len(chatbot.index.chunks),
            "embedder": chatbot.index.embedder,
            "answerer": settings.answerer,
        }

    @app.get("/sources")
    def sources() -> list[dict[str, str | int]]:
        return chatbot.index.documents()

    @app.post("/ask")
    def ask(body: AskRequest) -> AskResponse:
        started = time.perf_counter()
        answer = chatbot.ask(body.question, body.top_k)
        return AskResponse(
            answer=answer.text,
            refused=answer.refused,
            citations=[CitationOut(**asdict(citation)) for citation in answer.citations],
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    @app.get("/", include_in_schema=False)
    def index_page() -> HTMLResponse:
        return HTMLResponse(page)

    return app
