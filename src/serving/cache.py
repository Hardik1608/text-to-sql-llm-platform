"""
Query cache — Redis in production, in-memory for local runs and tests.

Generation is deterministic (temperature 0), so the key covers everything that can change
the answer: question, database, mode, model/adapter version and prompt template version.
Redis failures degrade to a cache miss; they never fail a request.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Protocol


class CacheError(Exception):
    pass


class QueryCache(Protocol):
    async def get(self, key: str) -> dict | None: ...
    async def set(self, key: str, value: dict, ttl_s: int) -> None: ...
    async def ping(self) -> bool: ...
    async def aclose(self) -> None: ...


def make_cache_key(*, question: str, db_id: str, mode: str, model: str, prompt_version: str) -> str:
    normalized = " ".join(question.split()).rstrip(" ?.!")
    raw = json.dumps([normalized, db_id, mode, model, prompt_version], ensure_ascii=False)
    return "t2s:v1:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RedisCache:
    def __init__(self, url: str, socket_timeout_s: float = 0.5):
        import redis.asyncio as redis  # imported lazily: optional at runtime

        self._redis = redis.from_url(
            url,
            socket_timeout=socket_timeout_s,
            socket_connect_timeout=socket_timeout_s,
            decode_responses=True,
        )

    async def get(self, key: str) -> dict | None:
        try:
            raw = await self._redis.get(key)
        except Exception as e:  # redis.RedisError, OSError, timeouts
            raise CacheError(str(e)) from e
        return json.loads(raw) if raw else None

    async def set(self, key: str, value: dict, ttl_s: int) -> None:
        try:
            await self._redis.set(key, json.dumps(value, default=str), ex=ttl_s)
        except Exception as e:
            raise CacheError(str(e)) from e

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._redis.aclose()


class InMemoryCache:
    """Bounded LRU with TTL — for tests and single-process local runs."""

    def __init__(self, max_items: int = 10_000):
        self.max_items = max_items
        self._data: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    async def get(self, key: str) -> dict | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at < time.monotonic():
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    async def set(self, key: str, value: dict, ttl_s: int) -> None:
        self._data[key] = (time.monotonic() + ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        self._data.clear()
