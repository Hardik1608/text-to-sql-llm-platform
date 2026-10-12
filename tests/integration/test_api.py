import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.factory import build_pipeline
from tests.conftest import FakeLLM


@pytest.fixture()
def client(settings, pipeline):
    with TestClient(create_app(settings, pipeline)) as c:
        yield c


def test_health_ready_metrics(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["llm"] is True
    client.post(
        "/query", json={"question": "How many singers do we have?", "db_id": "concert_singer"}
    )
    body = client.get("/metrics").text
    assert "text2sql_requests_total" in body and "text2sql_build_info" in body


def test_query_ok(client):
    response = client.post(
        "/query",
        json={
            "question": "What are the names of the singers from France?",
            "db_id": "concert_singer",
            "mode": "rag",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["columns"] == ["Name"]
    assert sorted(r[0] for r in body["rows"]) == ["Justin Brown", "Rose White"]


def test_query_validation_and_errors(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 422  # too short
    assert client.post("/query", json={"question": "How many?", "mode": "magic"}).status_code == 422
    assert client.post("/query", json={"question": "How many?", "db_id": "nope"}).status_code == 404
    assert client.post("/query", json={"question": "How many?", "db_id": "concert_singer",
                                       "mode": "qlora", "adapter": "spider-v9"}).status_code == 400  # fmt: skip


def test_llm_outage_returns_503(settings):
    llm = FakeLLM()
    llm.fail_with = httpx.ConnectError("down")
    pipeline = build_pipeline(settings, llm=llm.client(max_retries=0), cache=None)
    with TestClient(create_app(settings, pipeline)) as c:
        assert c.get("/ready").status_code == 503
        response = c.post(
            "/query", json={"question": "How many singers?", "db_id": "concert_singer"}
        )
        assert response.status_code == 503 and response.json()["status"] == "llm_error"


def test_databases(client):
    assert client.get("/databases").json()["databases"] == ["concert_singer", "pets_1"]
    assert "CREATE TABLE singer" in client.get("/databases/concert_singer").text
    assert client.get("/databases/nope").status_code == 404


def test_adapter_admin(client):
    assert client.get("/adapters").json()["default"] == "spider-v1"
    payload = {"name": "spider-v2", "path": "/adapters/spider-v2"}
    assert client.post("/adapters/load", json=payload).status_code == 401
    assert (
        client.post("/adapters/load", json=payload, headers={"X-Admin-Token": "wrong"}).status_code
        == 401
    )
    ok = client.post("/adapters/load", json=payload, headers={"X-Admin-Token": "secret-token"})
    assert ok.status_code == 200
    assert [a["name"] for a in client.get("/adapters").json()["adapters"]] == [
        "spider-v1",
        "spider-v2",
    ]
    response = client.post(
        "/query",
        json={
            "question": "How many singers do we have?",
            "db_id": "concert_singer",
            "mode": "qlora",
            "adapter": "spider-v2",
        },
    )
    assert response.json()["adapter"] == "spider-v2"
    gone = client.post(
        "/adapters/unload", json={"name": "spider-v2"}, headers={"X-Admin-Token": "secret-token"}
    )
    assert gone.status_code == 200


def test_admin_disabled_without_token(settings, pipeline):
    app = create_app(settings.model_copy(update={"admin_token": None}), pipeline)
    with TestClient(app) as c:
        r = c.post(
            "/adapters/load",
            json={"name": "x", "path": "/x"},
            headers={"X-Admin-Token": "anything"},
        )
        assert r.status_code == 403


def test_names_are_trimmed(client):
    assert "CREATE TABLE singer" in client.get("/databases/concert_singer%20").text
    body = {"question": "How many singers do we have?", "db_id": " concert_singer ", "mode": "qlora",
            "adapter": "spider-v1 "}  # fmt: skip
    response = client.post("/query", json=body)
    assert response.status_code == 200 and response.json()["db_id"] == "concert_singer"


def test_web_ui_and_info(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "<title>Text-to-SQL</title>" in page.text
    for asset in ("/static/app.js", "/static/styles.css"):
        assert client.get(asset).status_code == 200
    info = client.get("/info").json()
    assert info["modes"] == ["prompt", "rag", "qlora", "qlora_rag"] and "version" in info


def test_compare_grades_benchmark_questions(settings, fake_llm):
    fake_llm.by_model["spider-v1"] = "SELECT count(*) FROM stadium"  # the adapter gets it wrong
    pipeline = build_pipeline(settings, llm=fake_llm.client(max_retries=0), cache=None)
    with TestClient(create_app(settings, pipeline)) as c:
        body = c.post(
            "/compare", json={"question": "How many singers do we have?", "db_id": "concert_singer"}
        ).json()
    assert body["gold"]["rows"] == [[4]] and body["gold"]["split"] == "dev"
    verdict = {r["mode"]: r["correct"] for r in body["results"]}
    assert verdict == {"prompt": True, "rag": True, "qlora": False, "qlora_rag": False}
    assert body["correct_count"] == 2
    assert sorted(map(sorted, body["agreement"])) == [["prompt", "rag"], ["qlora", "qlora_rag"]]


def test_compare_without_gold_reports_agreement(client):
    body = client.post(
        "/compare", json={"question": "Count the singers please", "db_id": "concert_singer"}
    ).json()
    assert body["gold"] is None and body["correct_count"] is None
    assert all(r["correct"] is None for r in body["results"])
    assert body["agreement"] == [["prompt", "rag", "qlora", "qlora_rag"]]  # same default answer


def test_compare_marks_missing_adapter_unavailable(settings, fake_llm):
    no_adapter = settings.model_copy(update={"adapters": ""})
    pipeline = build_pipeline(no_adapter, llm=fake_llm.client(max_retries=0), cache=None)
    with TestClient(create_app(no_adapter, pipeline)) as c:
        body = c.post(
            "/compare", json={"question": "How many singers do we have?", "db_id": "concert_singer"}
        ).json()
    status = {r["mode"]: r["status"] for r in body["results"]}
    assert status["qlora"] == status["qlora_rag"] == "unavailable" and status["prompt"] == "ok"


def test_benchmark_questions_endpoint(client):
    body = client.get("/databases/concert_singer/questions?limit=2").json()
    assert len(body["questions"]) == 2 and body["questions"][0]["split"] == "dev"
    assert client.get("/databases/nope/questions").status_code == 404
    assert (
        client.post("/compare", json={"question": "How many?", "db_id": "nope"}).status_code == 404
    )
