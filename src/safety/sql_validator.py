"""
SQL Safety Layer — validation pipeline for LLM-generated SQL.

LLM output is untrusted input. Every generated query passes these stages
before it may touch a database:

  Stage 1  AST parse        : sqlglot must parse it, and it must be exactly ONE statement
                              (blocks stacked queries such as `SELECT 1; DROP TABLE t`).
  Stage 2  Read-only        : the root must be a SELECT / UNION / INTERSECT / EXCEPT, and no
                              write/DDL/session node may appear anywhere in the tree — this
                              catches DML hidden inside CTEs, SELECT ... INTO, FOR UPDATE,
                              PRAGMA, ATTACH. Dangerous functions (load_extension, sleep,
                              file readers) are denied by name.
  Stage 3  Schema allowlist : every table must exist in the target database (or be a CTE),
                              and every column must resolve against the schema
                              (hallucinated identifiers are caught before execution).
  Stage 4  Execution guard  : enforced by `src.safety.executor.SafeSQLiteExecutor` —
                              read-only connection, SQLite authorizer, timeout, row cap.

The validator returns `safe_sql`, the re-serialized AST. The executor runs exactly the
statement that was validated, which closes parser-differential tricks where the validator
and the database would interpret the raw text differently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError, TokenError
from sqlglot.optimizer.qualify import qualify

from src.schema import DbSchema

# sqlglot logs a warning whenever it falls back to an opaque `Command` node; those
# statements are rejected below, so the warning is just noise in the service logs.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

# Node types that must never appear anywhere in a read-only query. Looked up by name so
# the list keeps working across sqlglot versions that add/rename expression classes.
_FORBIDDEN_NODE_NAMES = (
    "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "TruncateTable",
    "Command", "Pragma", "Attach", "Detach", "Transaction", "Commit", "Rollback",
    "Set", "Use", "Grant", "Revoke", "Into", "Copy", "Analyze", "Cache", "Uncache",
    "LoadData", "Kill", "Lock", "Describe", "Show",
)  # fmt: skip
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name) for name in _FORBIDDEN_NODE_NAMES if hasattr(exp, name)
)

ALLOWED_ROOTS: tuple[type[exp.Expression], ...] = (exp.Select, exp.SetOperation)

DEFAULT_FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {
        # SQLite: extension loading, file access, large allocations
        "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
        "zeroblob", "randomblob",
        # PostgreSQL / MySQL / SQL Server: sleeps, file + network access, process control
        "sleep", "benchmark", "pg_sleep", "pg_sleep_for", "pg_sleep_until",
        "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
        "lo_import", "lo_export", "dblink", "dblink_exec", "load_file", "xp_cmdshell",
        "pg_terminate_backend", "pg_cancel_backend", "set_config", "pg_reload_conf",
    }
)  # fmt: skip

# Violation codes (also the `reason` label of text2sql_validation_failures_total)
EMPTY = "empty"
TOO_LONG = "too_long"
PARSE_ERROR = "parse_error"
MULTIPLE_STATEMENTS = "multiple_statements"
NOT_READ_ONLY = "not_read_only"
FORBIDDEN_FUNCTION = "forbidden_function"
UNKNOWN_TABLE = "unknown_table"
UNKNOWN_COLUMN = "unknown_column"

HALLUCINATION_CODES = frozenset({UNKNOWN_TABLE, UNKNOWN_COLUMN})
UNSAFE_CODES = frozenset({MULTIPLE_STATEMENTS, NOT_READ_ONLY, FORBIDDEN_FUNCTION})


@dataclass(frozen=True)
class Violation:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass
class ValidationResult:
    sql: str
    safe_sql: str | None = None
    violations: list[Violation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations and self.safe_sql is not None

    @property
    def codes(self) -> list[str]:
        return [v.code for v in self.violations]

    def feedback(self) -> str:
        """Human-readable summary, fed back to the LLM by the repair loop."""
        return "; ".join(v.message for v in self.violations)


class SQLValidator:
    """Static validation of a single LLM-generated query against a database schema."""

    def __init__(
        self,
        dialect: str = "sqlite",
        check_columns: bool = True,
        forbidden_functions: frozenset[str] = DEFAULT_FORBIDDEN_FUNCTIONS,
        max_length: int = 10_000,
    ):
        self.dialect = dialect
        self.check_columns = check_columns
        self.forbidden_functions = forbidden_functions
        self.max_length = max_length

    def validate(self, sql: str, schema: DbSchema | None = None) -> ValidationResult:
        result = ValidationResult(sql=sql)
        text = (sql or "").strip()

        # ---- Stage 1: parse, exactly one statement --------------------------------
        if not text:
            return self._fail(result, EMPTY, "The query is empty.")
        if len(text) > self.max_length:
            return self._fail(result, TOO_LONG, f"The query exceeds {self.max_length} characters.")
        try:
            statements = [s for s in sqlglot.parse(text, read=self.dialect) if s is not None]
        except (ParseError, TokenError) as e:
            return self._fail(result, PARSE_ERROR, f"SQL syntax error: {_first_line(e)}")
        if not statements:
            return self._fail(result, EMPTY, "The query is empty.")
        if len(statements) > 1:
            return self._fail(
                result,
                MULTIPLE_STATEMENTS,
                f"Expected exactly one statement, got {len(statements)}.",
            )
        root = statements[0]
        while isinstance(root, exp.Subquery | exp.Paren) and root.this is not None:
            root = root.this  # "(SELECT ...)" -> "SELECT ..."

        # ---- Stage 2: read-only ----------------------------------------------------
        if not isinstance(root, ALLOWED_ROOTS):
            return self._fail(
                result,
                NOT_READ_ONLY,
                f"Only SELECT queries are allowed (got {root.key.upper()}).",
            )
        for node in root.walk():
            if isinstance(node, FORBIDDEN_NODES):
                return self._fail(
                    result,
                    NOT_READ_ONLY,
                    f"Write/DDL/session operation '{node.key.upper()}' is not allowed.",
                )
        for func in root.find_all(exp.Func):
            name = _function_name(func)
            if name in self.forbidden_functions:
                return self._fail(result, FORBIDDEN_FUNCTION, f"Function '{name}' is not allowed.")

        # ---- Stage 3: schema allowlist ---------------------------------------------
        cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
        referenced: list[str] = []
        for table in root.find_all(exp.Table):
            if not isinstance(table.this, exp.Identifier):
                result.violations.append(
                    Violation(FORBIDDEN_FUNCTION, "Table-valued functions are not allowed.")
                )
                continue
            if table.catalog or (table.db and table.db.lower() != "main"):
                result.violations.append(
                    Violation(
                        UNKNOWN_TABLE,
                        f"Cross-database reference '{table.sql(dialect=self.dialect)}' is not allowed.",
                    )
                )
                continue
            name = table.name
            if name.lower() in cte_names:
                continue
            if schema is not None and schema.table(name) is None:
                result.violations.append(
                    Violation(UNKNOWN_TABLE, f"Table '{name}' does not exist in the schema.")
                )
            elif name.lower() not in {t.lower() for t in referenced}:
                referenced.append(schema.table(name).name if schema is not None else name)
        result.tables = referenced

        if schema is not None and self.check_columns and not result.violations:
            violation, warning = self._check_columns(root, schema)
            if violation:
                result.violations.append(violation)
            if warning:
                result.warnings.append(warning)

        if not result.violations:
            result.safe_sql = root.sql(dialect=self.dialect, comments=False)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_columns(
        self, root: exp.Expression, schema: DbSchema
    ) -> tuple[Violation | None, str | None]:
        """Resolve every column against the schema with sqlglot's qualifier."""
        tree = root.copy()
        known_columns = {c.lower() for t in schema.tables for c in t.column_names}
        # SQLite quirk (used throughout Spider's gold SQL): a double-quoted token that is not
        # a column is treated as a string literal, e.g. WHERE country = "France".
        for column in list(tree.find_all(exp.Column)):
            ident = column.this
            if (
                isinstance(ident, exp.Identifier)
                and ident.quoted
                and not column.table
                and ident.name.lower() not in known_columns
            ):
                column.replace(exp.Literal.string(ident.name))
        try:
            qualify(
                tree,
                schema=schema.sqlglot_schema(),
                dialect=self.dialect,
                validate_qualify_columns=True,
                quote_identifiers=False,
            )
        except OptimizeError as e:
            message = _first_line(e)
            lowered = message.lower()
            if "could not be resolved" in lowered or "unknown column" in lowered:
                return Violation(UNKNOWN_COLUMN, f"Unknown column: {message}"), None
            if "unknown table" in lowered:
                return Violation(UNKNOWN_TABLE, f"Unknown table: {message}"), None
            return None, f"column check skipped: {message}"
        except Exception as e:  # sqlglot limitations must never block a valid query
            return None, f"column check skipped: {type(e).__name__}: {_first_line(e)}"
        return None, None

    @staticmethod
    def _fail(result: ValidationResult, code: str, message: str) -> ValidationResult:
        result.violations.append(Violation(code, message))
        return result


def _function_name(func: exp.Func) -> str:
    if isinstance(func, exp.Anonymous):
        return str(func.name).lower()
    return func.sql_name().lower()


def _first_line(err: Exception) -> str:
    text = str(err).strip()
    return text.splitlines()[0] if text else type(err).__name__
