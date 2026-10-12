"""
Stage 4 of the safety layer: guarded execution on SQLite.

Defense in depth — even if a malicious statement slipped past the static validator,
the database layer still refuses to do anything but read:
  - the connection is opened read-only (URI `mode=ro`) with `PRAGMA query_only = ON`
  - a SQLite authorizer allows only SELECT / READ / FUNCTION / RECURSIVE actions and
    denies dangerous functions by name
  - a progress handler aborts any query that exceeds its time budget
  - result sets are capped at `max_rows`; string/blob sizes are capped via SQLITE_LIMIT_LENGTH
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from src.safety.sql_validator import DEFAULT_FORBIDDEN_FUNCTIONS

# Authorizer action codes (fallbacks for Python builds that don't export every constant)
_SQLITE_READ = getattr(sqlite3, "SQLITE_READ", 20)
_SQLITE_SELECT = getattr(sqlite3, "SQLITE_SELECT", 21)
_SQLITE_FUNCTION = getattr(sqlite3, "SQLITE_FUNCTION", 31)
_SQLITE_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)
_READ_ONLY_ACTIONS = frozenset({_SQLITE_READ, _SQLITE_SELECT, _SQLITE_RECURSIVE})

_UNSET = object()


class ExecutionError(Exception):
    """The database rejected or failed to run the query."""


class QueryTimeoutError(ExecutionError):
    """The query exceeded its time budget and was interrupted."""


@dataclass
class ExecutionResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_ms: float

    @property
    def row_count(self) -> int:
        return len(self.rows)


class SafeSQLiteExecutor:
    def __init__(
        self,
        timeout_s: float = 5.0,
        max_rows: int | None = 1000,
        max_value_length: int = 1_000_000,
        progress_steps: int = 1_000,
    ):
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self.max_value_length = max_value_length
        self.progress_steps = progress_steps

    def execute(
        self, db_path: str | Path, sql: str, max_rows: int | None = _UNSET
    ) -> ExecutionResult:  # type: ignore[assignment]
        """Run one read-only query. `max_rows=None` fetches everything (used by evaluation)."""
        limit = self.max_rows if max_rows is _UNSET else max_rows
        db_path = Path(db_path)
        if not db_path.exists():
            raise ExecutionError(f"Database file not found: {db_path.name}")

        conn = self._connect(db_path)
        deadline = time.monotonic() + self.timeout_s
        conn.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0, self.progress_steps
        )
        start = time.perf_counter()
        try:
            cursor = conn.execute(sql)
            columns = [d[0] for d in (cursor.description or [])]
            if limit is None:
                rows, truncated = cursor.fetchall(), False
            else:
                rows = cursor.fetchmany(limit + 1)
                truncated = len(rows) > limit
                rows = rows[:limit]
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                raise QueryTimeoutError(
                    f"Query exceeded the {self.timeout_s:.1f}s time limit"
                ) from e
            raise ExecutionError(str(e)) from e
        except (sqlite3.Error, sqlite3.Warning) as e:
            raise ExecutionError(str(e)) from e
        finally:
            conn.close()
        return ExecutionResult(
            columns=columns,
            rows=rows,
            truncated=truncated,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )

    def _connect(self, db_path: Path) -> sqlite3.Connection:
        uri = db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=1.0, check_same_thread=False)
        # Spider databases contain some invalid UTF-8; never crash on decoding
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        conn.execute("PRAGMA query_only = ON")
        if hasattr(conn, "setlimit"):  # Python 3.11+
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, self.max_value_length)
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()  # load schema pre-authorizer
        conn.set_authorizer(_authorizer)
        return conn


def _authorizer(action: int, arg1, arg2, db_name, trigger) -> int:
    if action in _READ_ONLY_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == _SQLITE_FUNCTION:
        name = (arg2 or "").lower()
        return sqlite3.SQLITE_DENY if name in DEFAULT_FORBIDDEN_FUNCTIONS else sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY
