"""JSON calls to OpenAI-compatible endpoints, with errors the operator can act on."""

from __future__ import annotations

from typing import Any

import httpx2 as httpx

from ragbot.errors import ProviderError


def open_client(
    base_url: str,
    api_key: str | None,
    timeout: float,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return httpx.Client(
        base_url=base_url.rstrip("/") + "/", headers=headers, timeout=timeout, transport=transport
    )


def post_json(client: httpx.Client, path: str, payload: dict[str, Any]) -> Any:
    url = f"{client.base_url}{path}"
    try:
        response = client.post(path, json=payload)
    except httpx.TimeoutException as exc:
        raise ProviderError(
            f"{url} did not answer in time. Check the endpoint or raise RAGBOT_TIMEOUT_SECONDS."
        ) from exc
    except httpx.HTTPError as exc:
        raise ProviderError(
            f"Could not reach {url} ({type(exc).__name__}). Check the base URL and your network."
        ) from exc
    if response.status_code >= 400:
        raise ProviderError(f"{url} answered HTTP {response.status_code}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError(f"{url} did not return JSON. Is it an OpenAI-compatible API?") from exc
