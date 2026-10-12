import pytest

from src.safety.sql_validator import SQLValidator


@pytest.fixture(scope="module")
def schema(data_dir):
    from src.schema import SchemaCatalog

    return SchemaCatalog.from_json(data_dir / "processed" / "schemas.json").get("concert_singer")


@pytest.fixture(scope="module")
def validator():
    return SQLValidator()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM singer",
        "select name, country from singer where age > 30 order by age desc limit 3",
        "SELECT T2.concert_Name, count(*) FROM singer_in_concert AS T1 JOIN concert AS T2 "
        "ON T1.concert_ID = T2.concert_ID GROUP BY T2.concert_ID",
        "SELECT Country FROM singer WHERE Age > 40 INTERSECT SELECT Country FROM singer WHERE Age < 30",
        "SELECT Name FROM singer WHERE Singer_ID NOT IN (SELECT Singer_ID FROM singer_in_concert)",
        "WITH old AS (SELECT Name, Age FROM singer WHERE Age > 40) SELECT Name FROM old",
        'SELECT Name FROM singer WHERE Country = "France"',  # SQLite double-quoted string literal
        "SELECT Name, count(*) AS cnt FROM singer GROUP BY Name ORDER BY cnt DESC",
        "SELECT name FROM SINGER",  # identifiers are case-insensitive
        "SELECT * FROM main.singer",
        "(SELECT Name FROM singer)",
        "SELECT Name FROM singer;",
    ],
)
def test_valid_queries_pass(validator, schema, sql):
    result = validator.validate(sql, schema)
    assert result.ok, result.violations
    assert result.safe_sql


@pytest.mark.parametrize(
    "sql, code",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("SELEC name FROM", "parse_error"),
        ("SELECT 1; DROP TABLE singer", "multiple_statements"),
        ("INSERT INTO singer VALUES (9, 'x', 'y', 1)", "not_read_only"),
        ("UPDATE singer SET Age = 1", "not_read_only"),
        ("DELETE FROM singer", "not_read_only"),
        ("DROP TABLE singer", "not_read_only"),
        ("CREATE TABLE x (a INT)", "not_read_only"),
        ("ALTER TABLE singer ADD COLUMN x INT", "not_read_only"),
        ("PRAGMA table_info(singer)", "not_read_only"),
        ("ATTACH DATABASE 'other.db' AS other", "not_read_only"),
        ("VACUUM", "not_read_only"),
        ("REPLACE INTO singer VALUES (1, 'a', 'b', 2)", "not_read_only"),
        ("BEGIN TRANSACTION", "not_read_only"),
        ("WITH d AS (DELETE FROM singer RETURNING *) SELECT * FROM d", "not_read_only"),
        ("SELECT * INTO backup FROM singer", "not_read_only"),
        ("SELECT load_extension('evil.so')", "forbidden_function"),
        ("SELECT randomblob(1000000000)", "forbidden_function"),
        ("SELECT * FROM pragma_table_info('singer')", "forbidden_function"),
        ("SELECT * FROM singers", "unknown_table"),
        ("SELECT * FROM other_db.singer", "unknown_table"),
        ("SELECT nam FROM singer", "unknown_column"),
        ("SELECT T1.bogus FROM singer AS T1", "unknown_column"),
    ],
)
def test_rejections(validator, schema, sql, code):
    result = validator.validate(sql, schema)
    assert not result.ok
    assert result.codes[0] == code, result.violations
    assert result.safe_sql is None


def test_too_long(schema):
    result = SQLValidator(max_length=50).validate(
        "SELECT Name FROM singer WHERE " + "Age > 1 AND " * 10 + "1", schema
    )
    assert result.codes == ["too_long"]


def test_without_schema_only_structural_checks(validator):
    assert validator.validate("SELECT * FROM anything_at_all").ok
    assert validator.validate("SELECT max(sleep) FROM t").ok  # a *column* named sleep is fine
    assert not validator.validate("DELETE FROM anything").ok


def test_safe_sql_is_reserialized_without_comments(validator, schema):
    result = validator.validate("SELECT Name /* hidden */ FROM singer -- trailing", schema)
    assert result.ok
    assert "hidden" not in result.safe_sql and "trailing" not in result.safe_sql


def test_reports_referenced_tables(validator, schema):
    sql = "SELECT T1.Name FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID = T2.Singer_ID"
    assert validator.validate(sql, schema).tables == ["singer", "singer_in_concert"]


def test_feedback_is_human_readable(validator, schema):
    result = validator.validate("SELECT * FROM singers", schema)
    assert "singers" in result.feedback()
