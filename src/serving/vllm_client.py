"""
vLLM serving client + LoRA adapter registry.

vLLM exposes an OpenAI-compatible API. One base model is loaded once; LoRA adapters are
selected *per request* by sending the adapter's name as the `model` field — a single GPU
serves many fine-tuned variants (multi-tenant serving, A/B tests between adapter versions).

With `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, adapters can be hot-swapped without a
restart through `POST /v1/load_lora_adapter` and `POST /v1/unload_lora_adapter`.

The client speaks plain OpenAI chat-completions, so the same code also runs against
Ollama or any OpenAI-compatible server for CPU-only development.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from src.generation.prompts import ADAPTER_MODES


class LLMError(Exception):
    """Generation failed. `kind` is one of: timeout, connection, http_error, bad_response."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class AdapterNotFoundError(KeyError):
    pass


@dataclass
class Generation:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    finish_reason: str | None = None


class OpenAICompatibleClient:
    """Async client for vLLM's OpenAI-compatible server (also Ollama / TGI / OpenAI)."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,
        )

    async def chat(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> Generation:
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,  # greedy decoding: SQL should be deterministic
        }
        if stop:
            payload["stop"] = stop
        start = time.perf_counter()
        data = await self._post_with_retries("/chat/completions", payload)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            usage = data.get("usage") or {}
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("bad_response", f"Unexpected response shape: {e}") from e
        return Generation(
            text=text,
            model=data.get("model", model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - start) * 1000,
            finish_reason=choice.get("finish_reason"),
        )

    async def list_models(self) -> list[str]:
        response = await self._request("GET", "/models")
        return [m["id"] for m in response.json().get("data", [])]

    async def health(self) -> bool:
        try:
            await self._request("GET", "/models")
            return True
        except LLMError:
            return False

    async def load_lora_adapter(self, name: str, path: str) -> None:
        await self._request(
            "POST", "/load_lora_adapter", json={"lora_name": name, "lora_path": path}
        )

    async def unload_lora_adapter(self, name: str) -> None:
        await self._request("POST", "/unload_lora_adapter", json={"lora_name": name})

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------

    async def _post_with_retries(self, path: str, payload: dict) -> dict:
        attempt = 0
        while True:
            try:
                return (await self._request("POST", path, json=payload)).json()
            except LLMError as e:
                retryable = e.kind == "connection" or (
                    e.kind == "http_error" and str(e).startswith(("HTTP 5", "HTTP 429"))
                )
                if not retryable or attempt >= self.max_retries:
                    raise
                attempt += 1
                await asyncio.sleep(0.25 * 2**attempt)

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as e:
            raise LLMError("timeout", f"LLM request timed out: {e!r}") from e
        except httpx.TransportError as e:
            raise LLMError(
                "connection", f"Cannot reach LLM backend at {self.base_url}: {e!r}"
            ) from e
        if response.status_code >= 400:
            raise LLMError("http_error", f"HTTP {response.status_code}: {response.text[:300]}")
        return response


# ----------------------------------------------------------------------
# Adapter registry
# ----------------------------------------------------------------------


@dataclass
class AdapterInfo:
    name: str
    path: str | None = None
    manifest: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "path": self.path, "manifest": self.manifest}


class AdapterRegistry:
    """Maps adapter versions (e.g. spider-v1, spider-v2) to vLLM model names.

    Base-model modes send the base model id; adapter modes send the adapter name, which vLLM
    resolves to the LoRA weights registered under that name (--lora-modules or runtime load).
    """

    def __init__(
        self,
        base_model: str,
        adapters: dict[str, str | None] | None = None,
        default: str | None = None,
    ):
        self.base_model = base_model
        self._adapters: dict[str, AdapterInfo] = {}
        self.default = default  # None -> the first registered adapter becomes the default
        for name, path in (adapters or {}).items():
            self.register(name, path)

    def register(self, name: str, path: str | None = None) -> AdapterInfo:
        info = AdapterInfo(name=name, path=path, manifest=_read_manifest(path))
        self._adapters[name] = info
        if self.default is None:
            self.default = name
        return info

    def unregister(self, name: str) -> None:
        self._adapters.pop(name, None)
        if self.default == name:
            self.default = next(iter(self._adapters), None)

    def list(self) -> list[AdapterInfo]:
        return list(self._adapters.values())

    def __contains__(self, name: str) -> bool:
        return name in self._adapters

    def resolve(self, mode: str, adapter: str | None = None) -> tuple[str, str | None]:
        """Return (model name to send to vLLM, adapter name or None for the base model)."""
        if mode not in ADAPTER_MODES:
            return self.base_model, None
        name = adapter or self.default
        if name is None:
            raise AdapterNotFoundError(
                f"Mode '{mode}' needs a LoRA adapter, but none is registered"
            )
        if name not in self._adapters:
            raise AdapterNotFoundError(f"Unknown adapter '{name}'")
        return name, name


def parse_adapters(spec: str) -> dict[str, str | None]:
    """'spider-v1=/adapters/spider-v1,spider-v2' -> {name: path_or_None}."""
    out: dict[str, str | None] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, _, path = item.partition("=")
        out[name.strip()] = path.strip() or None
    return out


def _read_manifest(path: str | None) -> dict:
    if not path:
        return {}
    manifest = Path(path) / "adapter_manifest.json"
    if not manifest.exists():
        return {}
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
