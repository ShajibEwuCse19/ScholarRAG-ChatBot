"""Minimal HTTP client for a local Ollama server."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

import requests

from src.config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_CONNECT_TIMEOUT,
    OLLAMA_MODEL,
    OLLAMA_READ_TIMEOUT,
    OLLAMA_TEMPERATURE,
)


class OllamaError(RuntimeError):
    """Raised for local Ollama connectivity or response failures."""


class OllamaClient:
    """Call Ollama's local chat API without a cloud dependency."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = session or requests.Session()
        self.timeout = (OLLAMA_CONNECT_TIMEOUT, OLLAMA_READ_TIMEOUT)

    def check_health(self) -> None:
        """Verify the server is reachable and the configured model exists."""

        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. Install/start Ollama first."
            ) from exc

        names = {str(item.get("name", "")) for item in payload.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise OllamaError(
                f"Ollama model '{self.model}' is not installed. Run: ollama pull {self.model}"
            )

    def stream_chat(self, messages: Sequence[dict[str, str]]) -> Iterator[str]:
        """Yield content deltas from Ollama's newline-delimited chat stream."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "options": {"temperature": OLLAMA_TEMPERATURE},
        }
        try:
            with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    try:
                        item = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise OllamaError("Ollama returned malformed streaming data") from exc
                    if item.get("error"):
                        raise OllamaError(str(item["error"]))
                    content = item.get("message", {}).get("content", "")
                    if content:
                        yield str(content)
        except requests.RequestException as exc:
            raise OllamaError(
                f"Ollama request failed at {self.base_url}: {exc}"
            ) from exc
