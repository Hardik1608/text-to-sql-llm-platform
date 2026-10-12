import pytest

from src.generation.postprocess import extract_sql
from src.generation.prompts import (
    CANNOT_ANSWER,
    FEW_SHOT_EXAMPLES,
    MODES,
    SYSTEM_PROMPT,
    build_prompt,
    format_sft_example,
)

SCHEMA = "CREATE TABLE singer (\n  Singer_ID NUMBER,\n  Name TEXT\n);"


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_sees_the_target_schema(mode):
    messages = build_prompt("How many singers?", SCHEMA, mode)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert SCHEMA in messages[1]["content"]
    assert messages[1]["content"].endswith("Question: How many singers?\nSQL:")


def test_prompt_mode_uses_static_synthetic_examples():
    content = build_prompt("q?", SCHEMA, "prompt")[1]["content"]
    assert all(ex["sql"] in content for ex in FEW_SHOT_EXAMPLES)
    assert "concert" not in content.lower()  # no Spider evaluation databases in prompts


def test_rag_mode_includes_examples_and_hints():
    content = build_prompt(
        "q?",
        SCHEMA,
        "rag",
        examples=[{"question": "How many cars?", "query": "SELECT count(*) FROM cars"}],
        value_hints=["singer.Name = 'Joe'"],
    )[1]["content"]
    assert "SELECT count(*) FROM cars" in content
    assert "-- singer.Name = 'Joe'" in content
    assert content.index("Similar examples") < content.index("Database Schema")


def test_qlora_mode_is_minimal():
    content = build_prompt("q?", SCHEMA, "qlora")[1]["content"]
    assert content == f"Database Schema:\n{SCHEMA}\n\nQuestion: q?\nSQL:"


def test_unknown_mode():
    with pytest.raises(ValueError):
        build_prompt("q", SCHEMA, "zero_shot")


def test_sft_example_is_prompt_completion():
    record = {
        "db_id": "x",
        "question": "How many singers?",
        "query": "SELECT count(*) FROM singer",
        "schema_sql": SCHEMA,
    }
    ex = format_sft_example(record)
    assert ex["completion"] == [{"role": "assistant", "content": record["query"]}]
    assert [m["role"] for m in ex["prompt"]] == ["system", "user"]
    assert ex["prompt"] == build_prompt(record["question"], SCHEMA, "qlora")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("SELECT count(*) FROM singer", "SELECT count(*) FROM singer"),
        ("SELECT count(*) FROM singer;", "SELECT count(*) FROM singer"),
        ("```sql\nSELECT name\nFROM singer\n```", "SELECT name\nFROM singer"),
        ("SQL: SELECT 1", "SELECT 1"),
        ("Here is the query:\nSELECT a FROM t\n\nThis query selects a.", "SELECT a FROM t"),
        ("SELECT a FROM t\nExplanation: picks a", "SELECT a FROM t"),
        ("WITH x AS (SELECT 1) SELECT * FROM x", "WITH x AS (SELECT 1) SELECT * FROM x"),
        ("CANNOT_ANSWER", CANNOT_ANSWER),
        ("I'm sorry, CANNOT_ANSWER.", CANNOT_ANSWER),
        ("", ""),
    ],
)
def test_extract_sql(raw, expected):
    assert extract_sql(raw) == expected


def test_extract_sql_keeps_injected_second_statement_visible():
    assert extract_sql("SELECT 1; DROP TABLE singer") == "SELECT 1; DROP TABLE singer"
