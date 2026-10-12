import asyncio

import httpx
import pytest
from prometheus_client import REGISTRY

from src.factory import build_pipeline
from src.pipeline import QueryRequest, UnknownDatabaseError
from src.serving.vllm_client import AdapterNotFoundError
from tests.conftest import FakeLLM


def run(pipeline, **kwargs):
    return asyncio.run(pipeline.run(QueryRequest(**kwargs)))


def metric(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_happy_path(pipeline):
    result = run(
        pipeline,
        question="What are the names of the singers from France?",
        db_id="concert_singer",
        mode="rag",
    )
    assert result.status == "ok"
    assert sorted(r[0] for r in result.rows) == ["Justin Brown", "Rose White"]
    assert result.model == "test-base" and result.adapter is None
    assert result.value_hints == ["singer.Country = 'France'"]
    assert {"retrieval_ms", "generation_ms", "validation_ms", "execution_ms", "total_ms"} <= set(
        result.timings
    )
    assert result.usage["prompt_tokens"] > 0


@pytest.mark.parametrize("mode", ["qlora", "qlora_rag"])
def test_adapter_modes_route_to_lora_model(pipeline, fake_llm, mode):
    result = run(
        pipeline, question="How many singers do we have?", db_id="concert_singer", mode=mode
    )
    assert result.status == "ok" and result.rows == [[4]]
    assert result.adapter == "spider-v1"
    assert fake_llm.requests[-1]["body"]["model"] == "spider-v1"


def test_unsafe_sql_is_rejected_without_repair(settings):
    llm = FakeLLM(default="SELECT 1; DROP TABLE singer")
    pipeline = build_pipeline(settings, llm=llm.client(max_retries=0), cache=None)
    before = metric("text2sql_validation_failures_total", reason="multiple_statements")
    result = run(
        pipeline, question="Delete every singer please", db_id="concert_singer", mode="prompt"
    )
    assert result.status == "rejected"
    assert result.violations[0]["code"] == "multiple_statements"
    assert result.repair_attempts == 0 and result.rows is None
    assert len(llm.requests) == 1
    assert metric("text2sql_validation_failures_total", reason="multiple_statements") == before + 1


def test_hallucinated_column_is_repaired(settings):
    question = "What are the names of the singers from France?"
    llm = FakeLLM(script={question: ["SELECT nam FROM singer WHERE Country = 'France'",
                                     "SELECT Name FROM singer WHERE Country = 'France'"]})  # fmt: skip
    pipeline = build_pipeline(settings, llm=llm.client(max_retries=0), cache=None)
    result = run(pipeline, question=question, db_id="concert_singer", mode="rag")
    assert result.status == "ok" and result.repaired and result.repair_attempts == 1
    repair_prompt = llm.requests[-1]["body"]["messages"][-1]["content"]
    assert "nam" in repair_prompt and "invalid" in repair_prompt


def test_execution_error_is_repaired(settings):
    question = "How many singers do we have?"
    llm = FakeLLM(script={question: ["SELECT count(*) FROM singer WHERE missing_fn(Age) > 1",
                                     "SELECT count(*) FROM singer"]})  # fmt: skip
    pipeline = build_pipeline(settings, llm=llm.client(max_retries=0), cache=None)
    result = run(pipeline, question=question, db_id="concert_singer", mode="prompt")
    assert result.status == "ok" and result.repaired and result.rows == [[4]]


def test_repair_budget_is_respected(settings):
    llm = FakeLLM(default="SELECT nam FROM singer")
    pipeline = build_pipeline(settings, llm=llm.client(max_retries=0), cache=None)
    result = run(pipeline, question="names?", db_id="concert_singer", mode="rag")
    assert result.status == "rejected" and result.repair_attempts == 1 and not result.repaired
    assert len(llm.requests) == 2


def test_cannot_answer(settings):
    pipeline = build_pipeline(settings, llm=FakeLLM(default="CANNOT_ANSWER").client(), cache=None)
    result = run(
        pipeline, question="What is the weather today?", db_id="concert_singer", mode="rag"
    )
    assert result.status == "cannot_answer" and result.rows is None


def test_llm_outage_is_reported(settings):
    llm = FakeLLM()
    llm.fail_with = httpx.ConnectError("down")
    pipeline = build_pipeline(settings, llm=llm.client(max_retries=0), cache=None)
    before = metric("text2sql_llm_errors_total", kind="connection")
    result = run(
        pipeline, question="How many singers do we have?", db_id="concert_singer", mode="rag"
    )
    assert result.status == "llm_error" and "Cannot reach" in result.error
    assert metric("text2sql_llm_errors_total", kind="connection") == before + 1


def test_cache_skips_generation_but_reexecutes(pipeline, fake_llm):
    q = "How many concerts were held in 2014?"
    first = run(pipeline, question=q, db_id="concert_singer", mode="rag")
    n_calls = len(fake_llm.requests)
    second = run(pipeline, question=q, db_id="concert_singer", mode="rag")
    assert not first.cached and second.cached
    assert len(fake_llm.requests) == n_calls  # no LLM call on a hit
    assert second.rows == first.rows == [[2]]
    assert "generation_ms" not in second.timings and "execution_ms" in second.timings
    bypass = run(pipeline, question=q, db_id="concert_singer", mode="rag", use_cache=False)
    assert not bypass.cached and len(fake_llm.requests) == n_calls + 1


def test_routing_without_db_id(pipeline):
    result = run(pipeline, question="How many pets are there?", mode="rag")
    assert result.routed and result.db_id == "pets_1" and result.rows == [[3]]


def test_execute_false_returns_validated_sql_only(pipeline):
    result = run(
        pipeline, question="How many singers do we have?", db_id="concert_singer", execute=False
    )
    assert (
        result.status == "ok"
        and result.sql == "SELECT COUNT(*) FROM singer"
        and result.rows is None
    )


def test_bad_requests_raise(pipeline, settings):
    with pytest.raises(UnknownDatabaseError):
        run(pipeline, question="q?", db_id="nope")
    with pytest.raises(ValueError):
        run(pipeline, question="q?", db_id="concert_singer", mode="zero_shot")
    no_adapter = build_pipeline(
        settings.model_copy(update={"adapters": ""}), llm=FakeLLM().client(), cache=None
    )
    with pytest.raises(AdapterNotFoundError):
        run(no_adapter, question="q?", db_id="concert_singer", mode="qlora")


def test_request_metrics(pipeline):
    before = metric("text2sql_requests_total", mode="prompt", status="ok")
    run(
        pipeline,
        question="How many singers do we have?",
        db_id="concert_singer",
        mode="prompt",
        use_cache=False,
    )
    assert metric("text2sql_requests_total", mode="prompt", status="ok") == before + 1
    assert metric("text2sql_request_latency_seconds_count", mode="prompt") >= 1
    assert metric("text2sql_inflight_requests") == 0
