"""Regression checks against the real Spider benchmark (skipped when data/ is absent, e.g. in CI)."""

import json
from pathlib import Path

import pytest

from src.evaluation.metrics import classify_hardness, exact_match
from src.safety.executor import SafeSQLiteExecutor
from src.safety.sql_validator import SQLValidator
from src.schema import SchemaCatalog

DATA = Path("data")
pytestmark = [
    pytest.mark.spider,
    pytest.mark.skipif(
        not (DATA / "processed" / "dev.jsonl").exists(), reason="Spider not downloaded"
    ),
]


@pytest.fixture(scope="module")
def dev():
    with open(DATA / "processed" / "dev.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def catalog():
    raw = DATA / "raw" / "spider_data"
    return SchemaCatalog.from_json(
        DATA / "processed" / "schemas.json", [raw / "database", raw / "test_database"]
    )


def test_validator_accepts_every_dev_gold_query(dev, catalog):
    validator = SQLValidator()
    rejected = [
        r["query"] for r in dev if not validator.validate(r["query"], catalog.get(r["db_id"])).ok
    ]
    assert rejected == []


def test_reserialized_sql_is_equivalent(dev, catalog):
    validator, executor = SQLValidator(), SafeSQLiteExecutor(timeout_s=30, max_rows=None)
    for r in dev[:200]:
        res = validator.validate(r["query"], catalog.get(r["db_id"]))
        db = catalog.db_path(r["db_id"])
        assert sorted(map(repr, executor.execute(db, r["query"]).rows)) == sorted(
            map(repr, executor.execute(db, res.safe_sql).rows)
        )
        assert exact_match(res.safe_sql, r["query"], catalog.get(r["db_id"]))


def test_hardness_distribution_tracks_official_split(dev):
    counts = {level: 0 for level in ("easy", "medium", "hard", "extra")}
    for r in dev:
        counts[classify_hardness(r["query"])] += 1
    official = {"easy": 248, "medium": 446, "hard": 174, "extra": 166}
    assert all(abs(counts[k] - official[k]) <= 20 for k in official), counts
