from __future__ import annotations

import json
from collections.abc import Callable

import httpx2 as httpx
import pytest

from ragbot.errors import ProviderError
from ragbot.llm import INSUFFICIENT, OpenAIChatAnswerer

Handler = Callable[[httpx.Request], httpx.Response]


def answerer(handler: Handler) -> OpenAIChatAnswerer:
    return OpenAIChatAnswerer(
        base_url="https://llm.test/v1",
        model="test-model",
        api_key="secret-key",
        transport=httpx.MockTransport(handler),
    )


def chat_reply(content: str) -> httpx.Response:
    body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    return httpx.Response(200, json=body)


def test_request_carries_model_key_and_numbered_sources(make_passage):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return chat_reply("The fee is 0.25% [1].")

    answer = answerer(handler).answer("What is the fee?", [make_passage("The swap fee is 0.25%.")])

    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["authorization"] == "Bearer secret-key"
    assert seen["body"]["model"] == "test-model"
    assert "[1] Doc > Section\nThe swap fee is 0.25%." in seen["body"]["messages"][1]["content"]
    assert answer.text == "The fee is 0.25% [1]."
    assert answer.citations[0].source == "doc.md"


def test_citation_number_the_model_invented_is_removed(make_passage):
    reply = chat_reply("The fee is 0.25% [1]. It doubles on Sundays [5].")
    answer = answerer(lambda _request: reply).answer("fee?", [make_passage("The fee is 0.25%.")])
    assert answer.text == "The fee is 0.25% [1]. It doubles on Sundays."
    assert [citation.n for citation in answer.citations] == [1]


@pytest.mark.parametrize("content", [INSUFFICIENT, "The fee is probably around 1%.", ""])
def test_reply_without_grounding_becomes_a_refusal(make_passage, content):
    answer = answerer(lambda _request: chat_reply(content)).answer("fee?", [make_passage("x")])
    assert answer.refused
    assert answer.citations == ()


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_error_names_the_status_and_never_the_key(make_passage, status):
    reply = httpx.Response(status, json={"error": {"message": "upstream said no"}})
    with pytest.raises(ProviderError) as raised:
        answerer(lambda _request: reply).answer("fee?", [make_passage("x")])
    assert f"HTTP {status}" in str(raised.value)
    assert "upstream said no" in str(raised.value)
    assert "secret-key" not in str(raised.value)


def test_timeout_names_the_setting_to_change(make_passage):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(ProviderError, match="RAGBOT_TIMEOUT_SECONDS"):
        answerer(handler).answer("fee?", [make_passage("x")])


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"text": "old API"}]}])
def test_reply_in_an_unexpected_shape_is_reported(make_passage, body):
    reply = httpx.Response(200, json=body)
    with pytest.raises(ProviderError, match="OpenAI-compatible"):
        answerer(lambda _request: reply).answer("fee?", [make_passage("x")])
