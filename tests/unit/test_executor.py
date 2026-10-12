import sqlite3

import pytest

from src.safety.executor import ExecutionError, QueryTimeoutError, SafeSQLiteExecutor


@pytest.fixture(scope="module")
def db_path(data_dir):
    return (
        data_dir / "raw" / "spider_data" / "database" / "concert_singer" / "concert_singer.sqlite"
    )


def test_select_returns_rows(db_path):
    out = SafeSQLiteExecutor().execute(db_path, "SELECT Name, Age FROM singer ORDER BY Age DESC")
    assert out.columns == ["Name", "Age"]
    assert out.rows[0] == ("Joe Sharp", 52)
    assert out.row_count == 4 and not out.truncated


def test_row_cap_and_unlimited(db_path):
    capped = SafeSQLiteExecutor(max_rows=2).execute(db_path, "SELECT * FROM singer")
    assert capped.row_count == 2 and capped.truncated
    everything = SafeSQLiteExecutor(max_rows=2).execute(
        db_path, "SELECT * FROM singer", max_rows=None
    )
    assert everything.row_count == 4 and not everything.truncated


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM singer",
        "UPDATE singer SET Age = 0",
        "INSERT INTO singer VALUES (99, 'x', 'y', 1)",
        "DROP TABLE singer",
        "CREATE TABLE evil (a)",
        "ATTACH DATABASE ':memory:' AS other",
        "PRAGMA writable_schema = ON",
    ],
)
def test_writes_are_blocked_even_without_the_validator(db_path, sql):
    with pytest.raises(ExecutionError):
        SafeSQLiteExecutor().execute(db_path, sql)
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT count(*) FROM singer").fetchone()[0] == 4
    conn.close()


def test_dangerous_function_denied_by_authorizer(db_path):
    with pytest.raises(ExecutionError):
        SafeSQLiteExecutor().execute(db_path, "SELECT load_extension('evil')")


def test_runaway_query_is_interrupted(db_path):
    infinite = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    )
    with pytest.raises(QueryTimeoutError):
        SafeSQLiteExecutor(timeout_s=0.2).execute(db_path, infinite)


def test_bounded_recursive_cte_allowed(db_path):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 5) SELECT sum(x) FROM c"
    assert SafeSQLiteExecutor().execute(db_path, sql).rows == [(15,)]


def test_missing_database(tmp_path):
    with pytest.raises(ExecutionError):
        SafeSQLiteExecutor().execute(tmp_path / "nope.sqlite", "SELECT 1")


def test_syntax_error_is_execution_error(db_path):
    with pytest.raises(ExecutionError):
        SafeSQLiteExecutor().execute(db_path, "SELECT FROM WHERE")
