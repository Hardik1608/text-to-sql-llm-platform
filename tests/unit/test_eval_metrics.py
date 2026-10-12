import pytest

from src.evaluation.metrics import (
    EvalRecord,
    classify_hardness,
    exact_match,
    order_matters,
    results_match,
    summarize,
)


def test_results_match_semantics():
    assert results_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")])  # order-insensitive
    assert not results_match([(1,), (2,)], [(2,), (1,)], order_matters=True)
    assert results_match([("a", 1)], [(1, "a")])  # column permutation
    assert results_match([(1.0,)], [(1,)])  # int vs float
    assert results_match([], [])
    assert not results_match([(1,)], [])
    assert not results_match([(1,), (1,)], [(1,)])  # multiset, not set
    assert not results_match([(1, 2)], [(1,)])


def test_order_matters_follows_spider_convention():
    assert order_matters("SELECT a FROM t ORDER BY a")
    assert not order_matters("SELECT a FROM t")


def test_exact_match_normalization():
    assert exact_match(
        "SELECT T1.name FROM singer AS T1 WHERE T1.age > 3",
        "select name from singer where age > 40",
    )
    assert exact_match("SELECT count(*) FROM singer", "SELECT COUNT(*) FROM SINGER")
    assert not exact_match("SELECT name FROM singer", "SELECT country FROM singer")
    assert not exact_match(None, "SELECT 1")
    assert not exact_match("SELEC", "SELECT 1")


@pytest.mark.parametrize(
    "sql, level",
    [
        ("SELECT count(*) FROM singer", "easy"),
        ("SELECT name, country, age FROM singer ORDER BY age DESC", "medium"),
        (
            "SELECT country FROM singer WHERE age > 40 INTERSECT SELECT country FROM singer WHERE age < 30",
            "hard",
        ),
        (
            "SELECT T2.name, count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id "
            "WHERE T1.year >= 2014 GROUP BY T2.stadium_id ORDER BY count(*) DESC LIMIT 1",
            "extra",
        ),
    ],
)
def test_hardness(sql, level):
    assert classify_hardness(sql) == level


def _record(**kw):
    base = dict(
        idx=0, db_id="d", question="q", gold_sql="g", mode="rag", pred_sql="p", status="ok",
        exec_match=True, exact_match=False, hardness="easy", latency_ms=100.0, generation_ms=90.0,
        prompt_tokens=500, completion_tokens=20,
    )  # fmt: skip
    base.update(kw)
    return EvalRecord(**base)


def test_summarize():
    records = [
        _record(),
        _record(
            exec_match=False,
            status="rejected",
            violation_codes=["unknown_column"],
            latency_ms=300.0,
        ),
        _record(
            exec_match=False, status="rejected", violation_codes=["not_read_only"], hardness="hard"
        ),
        _record(exec_match=False, status="exec_error"),
    ]
    s = summarize(records)
    assert s["n"] == 4
    assert s["execution_accuracy"] == 0.25
    assert s["valid_sql_rate"] == 0.5
    assert s["hallucination_rate"] == 0.25
    assert s["unsafe_generated_rate"] == 0.25
    assert s["by_hardness"]["easy"] == {"n": 3, "execution_accuracy": 0.3333}
    assert s["latency_ms_p95"] > s["latency_ms_p50"]
