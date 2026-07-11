"""Gemini REST adapters and response parsing (injectable transport, no network in tests)."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol

from . import VidqaError

Transport = Callable[[str, str, dict[str, str], dict[str, Any], float], tuple[int, str]]


class ProviderError(VidqaError):
    pass


class Captioner(Protocol):
    name: str

    def caption(self, image_path: str, local_facts: dict[str, Any]) -> str: ...


def default_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float,
) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8") if body else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        raise ProviderError(f"Gemini request failed: {exc.reason}") from exc


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ProviderError("response did not contain a JSON object")
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ProviderError(f"invalid JSON response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("response JSON must be an object")
    return parsed


def image_part(path: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    return {
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(raw).decode("ascii"),
        }
    }


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        transport: Transport | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is not set")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
        self.base_url = (
            base_url or os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        ).rstrip("/")
        self.transport = transport or default_transport
        self.timeout = timeout

    def generate(self, parts: list[dict[str, Any]]) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent"
        body = {"contents": [{"role": "user", "parts": parts}]}
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        status, text = self.transport("POST", url, headers, body, self.timeout)
        if status >= 400:
            raise ProviderError(f"Gemini HTTP {status}: {text[:400]}")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError("Gemini returned invalid JSON") from exc
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ProviderError("Gemini returned no candidates")
        response_parts = candidates[0].get("content", {}).get("parts", [])
        rendered = "".join(str(part.get("text", "")) for part in response_parts).strip()
        if not rendered:
            raise ProviderError("Gemini returned an empty response")
        return rendered


class GeminiCaptioner:
    name = "gemini"

    def __init__(self, client: GeminiClient | None = None, **kwargs: Any) -> None:
        self.client = client or GeminiClient(**kwargs)

    def caption(self, image_path: str, local_facts: dict[str, Any]) -> str:
        prompt = (
            "Caption this video keyframe in one or two sentences. State the dominant "
            "background colour, visible coloured shapes and approximate counts, and any "
            'motion implied by the scene. Return JSON exactly like {"caption": "..."}.'
        )
        rendered = self.client.generate([{"text": prompt}, image_part(image_path)])
        try:
            parsed = extract_json(rendered)
            caption = str(parsed.get("caption", "")).strip()
            if caption:
                return caption
        except ProviderError:
            pass
        return rendered.strip()
