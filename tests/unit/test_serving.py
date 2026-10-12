import asyncio

import httpx
import pytest

from src.serving.cache import InMemoryCache, make_cache_key
from src.serving.vllm_client import (
    AdapterNotFoundError,
    AdapterRegistry,
    LLMError,
    parse_adapters,
)
from tests.conftest import FakeLLM

MESSAGES = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "Question: How many?\nSQL:"},
]


def test_chat_parses_content_and_usage():
    llm = FakeLLM(script={"How many?": ["SELECT 1"]})
    gen = asyncio.run(llm.client().chat(MESSAGES, model="m"))
    assert gen.text == "SELECT 1" and gen.model == "m" and gen.prompt_tokens > 0
    body = llm.requests[0]["body"]
    assert body["temperature"] == 0.0 and body["max_tokens"] == 256


def test_retries_transient_5xx():
    llm = FakeLLM(script={"How many?": ["SELECT 1"]})
    llm.status_codes = [503]
    assert asyncio.run(llm.client(max_retries=1).chat(MESSAGES, model="m")).text == "SELECT 1"


def test_client_errors_are_not_retried():
    llm = FakeLLM()
    llm.status_codes = [400, 200]
    with pytest.raises(LLMError) as err:
        asyncio.run(llm.client(max_retries=3).chat(MESSAGES, model="m"))
    assert err.value.kind == "http_error" and len(llm.requests) == 1


@pytest.mark.parametrize(
    "exc, kind",
    [(httpx.ConnectError("refused"), "connection"), (httpx.ReadTimeout("slow"), "timeout")],
)
def test_transport_failures(exc, kind):
    llm = FakeLLM()
    llm.fail_with = exc
    with pytest.raises(LLMError) as err:
        asyncio.run(llm.client(max_retries=0).chat(MESSAGES, model="m"))
    assert err.value.kind == kind


def test_health_and_adapter_hot_swap():
    llm = FakeLLM()
    client = llm.client()

    async def scenario():
        assert await client.health()
        assert await client.list_models() == ["test-base", "spider-v1"]
        await client.load_lora_adapter("spider-v2", "/adapters/spider-v2")
        await client.unload_lora_adapter("spider-v1")

    asyncio.run(scenario())
    paths = [r["path"] for r in llm.requests]
    assert paths[-2:] == ["/v1/load_lora_adapter", "/v1/unload_lora_adapter"]
    assert llm.requests[-2]["body"] == {
        "lora_name": "spider-v2",
        "lora_path": "/adapters/spider-v2",
    }


def test_adapter_registry():
    registry = AdapterRegistry("base-model", parse_adapters("spider-v1=/a/v1, spider-v2"))
    assert registry.resolve("prompt") == ("base-model", None)
    assert registry.resolve("rag", "spider-v2") == (
        "base-model",
        None,
    )  # base modes ignore adapters
    assert registry.resolve("qlora") == ("spider-v1", "spider-v1")  # first registered is default
    assert registry.resolve("qlora_rag", "spider-v2") == ("spider-v2", "spider-v2")
    with pytest.raises(AdapterNotFoundError):
        registry.resolve("qlora", "spider-v9")
    registry.unregister("spider-v1")
    assert registry.default == "spider-v2"
    with pytest.raises(AdapterNotFoundError):
        AdapterRegistry("base-model").resolve("qlora")


def test_parse_adapters():
    assert parse_adapters("") == {}
    assert parse_adapters("a=/x,b") == {"a": "/x", "b": None}


def test_cache_key_semantics():
    key = make_cache_key(
        question="How many singers?", db_id="d", mode="rag", model="m", prompt_version="v2"
    )
    same = make_cache_key(
        question="  How many   singers ", db_id="d", mode="rag", model="m", prompt_version="v2"
    )
    assert key == same and key.startswith("t2s:v1:")
    for change in (
        {"mode": "qlora"},
        {"model": "spider-v2"},
        {"prompt_version": "v3"},
        {"db_id": "e"},
    ):
        args = {
            "question": "How many singers?",
            "db_id": "d",
            "mode": "rag",
            "model": "m",
            "prompt_version": "v2",
        }
        assert make_cache_key(**{**args, **change}) != key


def test_in_memory_cache_ttl_and_bound():
    cache = InMemoryCache(max_items=2)

    async def scenario():
        await cache.set("a", {"v": 1}, ttl_s=60)
        await cache.set("b", {"v": 2}, ttl_s=60)
        await cache.set("c", {"v": 3}, ttl_s=60)  # evicts "a"
        assert await cache.get("a") is None and await cache.get("c") == {"v": 3}
        await cache.set("d", {"v": 4}, ttl_s=-1)  # already expired
        assert await cache.get("d") is None

    asyncio.run(scenario())
