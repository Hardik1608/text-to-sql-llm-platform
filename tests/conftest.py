"""
Shared fixtures: a miniature Spider dataset (real SQLite files, Spider's JSON formats) and a
fake OpenAI-compatible LLM backend, so the whole platform runs in CI without a GPU, network
access or the 200 MB benchmark download.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import httpx
import pytest

from src.config import Settings
from src.ingestion import download_spider

# ----------------------------------------------------------------------
# Mini Spider
# ----------------------------------------------------------------------

MINI_DBS = {
    "concert_singer": {
        "tables": {
            "stadium": [("Stadium_ID", "number"), ("Name", "text"), ("Capacity", "number")],
            "singer": [("Singer_ID", "number"), ("Name", "text"), ("Country", "text"), ("Age", "number")],
            "concert": [("concert_ID", "number"), ("concert_Name", "text"), ("Stadium_ID", "number"), ("Year", "text")],
            "singer_in_concert": [("concert_ID", "number"), ("Singer_ID", "number")],
        },
        "pks": [("stadium", "Stadium_ID"), ("singer", "Singer_ID"), ("concert", "concert_ID")],
        "fks": [
            ("concert", "Stadium_ID", "stadium", "Stadium_ID"),
            ("singer_in_concert", "concert_ID", "concert", "concert_ID"),
            ("singer_in_concert", "Singer_ID", "singer", "Singer_ID"),
        ],
        "rows": {
            "stadium": [(1, "Stark's Park", 10104), (2, "Glebe Park", 4125), (3, "Somerset Park", 11998)],
            "singer": [
                (1, "Joe Sharp", "Netherlands", 52),
                (2, "Timbaland", "United States", 32),
                (3, "Justin Brown", "France", 29),
                (4, "Rose White", "France", 41),
            ],
            "concert": [(1, "Auditions", 1, "2014"), (2, "Super bootcamp", 2, "2014"), (3, "Home Visits", 1, "2015")],
            "singer_in_concert": [(1, 2), (1, 3), (2, 3), (3, 1)],
        },
    },
    "pets_1": {
        "tables": {
            "student": [("StuID", "number"), ("LName", "text"), ("Age", "number"), ("city_code", "text")],
            "pets": [("PetID", "number"), ("PetType", "text"), ("pet_age", "number")],
            "has_pet": [("StuID", "number"), ("PetID", "number")],
        },
        "pks": [("student", "StuID"), ("pets", "PetID")],
        "fks": [("has_pet", "StuID", "student", "StuID"), ("has_pet", "PetID", "pets", "PetID")],
        "rows": {
            "student": [(1001, "Smith", 18, "BAL"), (1002, "Kim", 19, "HKG"), (1003, "Jones", 21, "WAS")],
            "pets": [(2001, "cat", 3), (2002, "dog", 2), (2003, "dog", 1)],
            "has_pet": [(1001, 2001), (1002, 2002), (1002, 2003)],
        },
    },
}  # fmt: skip

TRAIN = [
    ("pets_1", "How many pets are there?", "SELECT count(*) FROM pets"),
    ("pets_1", "How many students do we have?", "SELECT count(*) FROM student"),
    ("pets_1", "What is the average age of all students?", "SELECT avg(Age) FROM student"),
    ("pets_1", "Find the last names of students older than 20.", "SELECT LName FROM student WHERE Age > 20"),
    ("pets_1", "How many dogs are there?", "SELECT count(*) FROM pets WHERE PetType = 'dog'"),
    ("pets_1", "Which students have a dog? Give their last names.",
     "SELECT T1.LName FROM student AS T1 JOIN has_pet AS T2 ON T1.StuID = T2.StuID "
     "JOIN pets AS T3 ON T2.PetID = T3.PetID WHERE T3.PetType = 'dog'"),
]  # fmt: skip
TRAIN_OTHERS = [("pets_1", "List all pet types.", "SELECT DISTINCT PetType FROM pets")]
DEV = [
    ("concert_singer", "How many singers do we have?", "SELECT count(*) FROM singer"),
    ("concert_singer", "What are the names of the singers from France?", "SELECT Name FROM singer WHERE Country = 'France'"),
    ("concert_singer", "Show the name and capacity of every stadium ordered by capacity.",
     "SELECT Name, Capacity FROM stadium ORDER BY Capacity DESC"),
    ("concert_singer", "How many concerts were held in 2014?", "SELECT count(*) FROM concert WHERE Year = '2014'"),
]  # fmt: skip


def _spider_entry(db_id: str, spec: dict) -> dict:
    tables = list(spec["tables"])
    columns_original: list = [[-1, "*"]]
    types = ["text"]
    for t_idx, table in enumerate(tables):
        for col, col_type in spec["tables"][table]:
            columns_original.append([t_idx, col])
            types.append(col_type)

    def col_index(table: str, col: str) -> int:
        return columns_original.index([tables.index(table), col])

    return {
        "db_id": db_id,
        "table_names_original": tables,
        "table_names": [t.replace("_", " ") for t in tables],
        "column_names_original": columns_original,
        "column_names": [[t, c.lower().replace("_", " ")] for t, c in columns_original],
        "column_types": types,
        "primary_keys": [col_index(t, c) for t, c in spec["pks"]],
        "foreign_keys": [[col_index(t, c), col_index(rt, rc)] for t, c, rt, rc in spec["fks"]],
    }


def _create_sqlite(path: Path, spec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    sql_types = {"number": "INTEGER", "text": "TEXT"}
    pks = dict(spec["pks"])
    for table, cols in spec["tables"].items():
        defs = [f'"{c}" {sql_types[t]}' for c, t in cols]
        if table in pks:
            defs.append(f'PRIMARY KEY ("{pks[table]}")')
        for t, c, rt, rc in spec["fks"]:
            if t == table:
                defs.append(f'FOREIGN KEY ("{c}") REFERENCES "{rt}"("{rc}")')
        conn.execute(f'CREATE TABLE "{table}" ({", ".join(defs)})')
        marks = ", ".join("?" * len(cols))
        conn.executemany(f'INSERT INTO "{table}" VALUES ({marks})', spec["rows"][table])
    conn.commit()
    conn.close()


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    """A data/ directory laid out exactly like the real one, built from the mini Spider."""
    root = tmp_path_factory.mktemp("data")
    spider = root / "raw" / "spider_data"
    spider.mkdir(parents=True)
    (spider / "tables.json").write_text(
        json.dumps([_spider_entry(db, spec) for db, spec in MINI_DBS.items()]), encoding="utf-8"
    )
    for fname, rows in (
        ("train_spider.json", TRAIN),
        ("train_others.json", TRAIN_OTHERS),
        ("dev.json", DEV),
    ):
        payload = [{"db_id": d, "question": q, "query": s} for d, q, s in rows]
        (spider / fname).write_text(json.dumps(payload), encoding="utf-8")
    for db_id, spec in MINI_DBS.items():
        _create_sqlite(spider / "database" / db_id / f"{db_id}.sqlite", spec)
    download_spider.run(raw_dir=root / "raw", processed_dir=root / "processed")
    return root


@pytest.fixture()
def settings(data_dir: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=data_dir,
        llm_base_url="http://llm.test/v1",
        base_model="test-base",
        adapters="spider-v1",
        cache_enabled=True,
        redis_url="memory://",
        admin_token="secret-token",
        max_repair_attempts=1,
    )


# ----------------------------------------------------------------------
# Fake OpenAI-compatible LLM
# ----------------------------------------------------------------------

_QUESTION = re.compile(r"Question: (.*?)\nSQL:")


class FakeLLM:
    """Scripted chat-completions backend.

    `script` maps a question to the outputs to return in order (the last one repeats);
    unknown questions get `default`. Every request is recorded for assertions.
    """

    def __init__(
        self,
        script: dict[str, list[str]] | None = None,
        default: str = "SELECT count(*) FROM singer",
    ):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.default = default
        self.requests: list[dict] = []
        self.fail_with: Exception | None = None
        self.status_codes: list[int] = []  # served before normal responses, e.g. [503]
        self.by_model: dict[str, str] = {}  # model/adapter name -> fixed answer (overrides script)

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.fail_with is not None:
            raise self.fail_with
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.requests.append({"path": path, "body": body})
        if self.status_codes:
            return httpx.Response(self.status_codes.pop(0), text="upstream error")
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "test-base"}, {"id": "spider-v1"}]})
        if path.endswith("/load_lora_adapter") or path.endswith("/unload_lora_adapter"):
            return httpx.Response(200, text="Success")
        user = [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
        question = self._question(body["messages"])
        outputs = self.script.get(question)
        text = (outputs.pop(0) if len(outputs) > 1 else outputs[0]) if outputs else self.default
        text = self.by_model.get(body["model"], text)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-test",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(text) // 4},
            },
        )

    @staticmethod
    def _question(messages: list[dict]) -> str:
        first_user = next(m for m in messages if m["role"] == "user")["content"]
        questions = _QUESTION.findall(first_user)  # few-shot examples come first; target is last
        return questions[-1].strip() if questions else ""

    def client(self, **kwargs):
        from src.serving.vllm_client import OpenAICompatibleClient

        return OpenAICompatibleClient(
            base_url="http://llm.test/v1", transport=httpx.MockTransport(self.handler), **kwargs
        )


@pytest.fixture()
def fake_llm() -> FakeLLM:
    gold = {q: [s] for _, q, s in TRAIN + TRAIN_OTHERS + DEV}
    return FakeLLM(script=gold)


@pytest.fixture()
def pipeline(settings: Settings, fake_llm: FakeLLM):
    from src.factory import build_pipeline

    return build_pipeline(settings, llm=fake_llm.client(max_retries=0))
