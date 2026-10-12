"""
Value retrieval — link question phrases to database cell values.

"Which singers are from France?" → hint `singer.Country = 'France'`.

Literal mismatches ('france' vs 'France', 'USA' vs 'United States', '2014' stored as TEXT)
are a major source of wrong-but-valid SQL. Retrieving the exact stored values that the
question mentions grounds the literals the model writes. This is retrieval over database
*content*, complementing BM25 retrieval over the schema.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass

from src.retrieval.text import STOPWORDS
from src.schema import SchemaCatalog, quote_ident

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&.\-]*")
_QUOTED = re.compile(r"\"([^\"]+)\"|'([^']+)'")
_TEXT_TYPES = ("TEXT", "CHAR", "VARCHAR", "CLOB", "STRING", "TIME")


@dataclass(frozen=True)
class ValueHint:
    table: str
    column: str
    value: str

    def render(self) -> str:
        escaped = self.value.replace("'", "''")
        return f"{quote_ident(self.table)}.{quote_ident(self.column)} = '{escaped}'"


def _normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .,;:!?")


class ValueIndex:
    def __init__(
        self,
        catalog: SchemaCatalog,
        max_values_per_column: int = 2000,
        max_value_chars: int = 60,
        max_ngram: int = 4,
    ):
        self.catalog = catalog
        self.max_values_per_column = max_values_per_column
        self.max_value_chars = max_value_chars
        self.max_ngram = max_ngram
        self._cache: dict[str, dict[str, list[tuple[str, str, str]]]] = {}

    def match(
        self,
        question: str,
        db_id: str,
        tables: Iterable[str] | None = None,
        limit: int = 6,
    ) -> list[ValueHint]:
        index = self._index(db_id)
        if not index:
            return []
        allowed = {t.lower() for t in tables} if tables is not None else None

        candidates: list[str] = [_normalize(a or b) for a, b in _QUOTED.findall(question)]
        words = _WORD.findall(question)
        for n in range(min(self.max_ngram, len(words)), 0, -1):  # longest phrases first
            for i in range(len(words) - n + 1):
                phrase = _normalize(" ".join(words[i : i + n]))
                if n == 1 and (phrase in STOPWORDS or len(phrase) < 2):
                    continue
                candidates.append(phrase)

        hints: list[ValueHint] = []
        seen: set[tuple[str, str, str]] = set()
        for phrase in candidates:
            for table, column, value in index.get(phrase, ()):
                key = (table, column, value)
                if key in seen or (allowed is not None and table.lower() not in allowed):
                    continue
                seen.add(key)
                hints.append(ValueHint(table, column, value))
                if len(hints) >= limit:
                    return hints
        return hints

    def _index(self, db_id: str) -> dict[str, list[tuple[str, str, str]]]:
        if db_id in self._cache:
            return self._cache[db_id]
        index: dict[str, list[tuple[str, str, str]]] = {}
        db_path = self.catalog.db_path(db_id)
        if db_path is not None:
            conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            try:
                for table in self.catalog.get(db_id).tables:
                    for col in table.columns:
                        if not col.type.upper().startswith(_TEXT_TYPES):
                            continue
                        self._index_column(conn, table.name, col.name, index)
            finally:
                conn.close()
        self._cache[db_id] = index
        return index

    def _index_column(self, conn: sqlite3.Connection, table: str, column: str, index: dict) -> None:
        sql = (
            f"SELECT DISTINCT {quote_ident(column)} FROM {quote_ident(table)} "
            f"WHERE {quote_ident(column)} IS NOT NULL LIMIT {self.max_values_per_column}"
        )
        try:
            values = [r[0] for r in conn.execute(sql)]
        except sqlite3.Error:
            return  # schema/DB mismatch in a few Spider databases: skip the column
        for value in values:
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not (2 <= len(text) <= self.max_value_chars):
                continue
            if text.isdigit() and len(text) < 4:  # small numbers match too much noise
                continue
            index.setdefault(_normalize(text), []).append((table, column, text))
