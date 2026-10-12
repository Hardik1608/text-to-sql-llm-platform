"""
Structured database schema model.

One representation shared by every stage of the platform:
  - retrieval  : table/column names for BM25 schema pruning
  - prompting  : CREATE TABLE rendering (with PK/FK constraints)
  - safety     : table/column allowlists for the SQL validator
  - execution  : resolving a db_id to its SQLite file

Schemas come from Spider's tables.json (the benchmark) or from introspecting
any SQLite database (bring-your-own-database).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_SIMPLE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Spider column types -> SQL types used in the rendered CREATE TABLE statements
_SPIDER_TYPES = {
    "text": "TEXT",
    "number": "NUMBER",
    "time": "TIME",
    "boolean": "BOOLEAN",
    "others": "TEXT",
}


def quote_ident(name: str) -> str:
    """Quote an identifier only when needed (keeps prompts readable)."""
    if _SIMPLE_IDENT.match(name):
        return name
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class Column:
    name: str
    type: str = "TEXT"


@dataclass
class Table:
    name: str
    columns: list[Column]
    primary_key: list[str] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass(frozen=True)
class ForeignKey:
    table: str
    column: str
    ref_table: str
    ref_column: str


@dataclass
class DbSchema:
    db_id: str
    tables: list[Table]
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Lookups (SQLite identifiers are case-insensitive)
    # ------------------------------------------------------------------

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def table(self, name: str) -> Table | None:
        lowered = name.lower()
        for t in self.tables:
            if t.name.lower() == lowered:
                return t
        return None

    def column_map(self) -> dict[str, list[str]]:
        return {t.name: t.column_names for t in self.tables}

    def sqlglot_schema(self) -> dict[str, dict[str, str]]:
        """Schema in the nested-dict form sqlglot's optimizer expects."""
        return {t.name: {c.name: c.type for c in t.columns} for t in self.tables}

    def neighbors(self, table: str) -> set[str]:
        """Tables directly connected to `table` through a foreign key."""
        lowered = table.lower()
        out: set[str] = set()
        for fk in self.foreign_keys:
            if fk.table.lower() == lowered:
                out.add(fk.ref_table)
            elif fk.ref_table.lower() == lowered:
                out.add(fk.table)
        return out

    def subset(self, tables: Iterable[str]) -> DbSchema:
        """Schema restricted to `tables` (FKs kept only when both ends survive)."""
        keep = {t.lower() for t in tables}
        kept = [t for t in self.tables if t.name.lower() in keep]
        fks = [
            fk
            for fk in self.foreign_keys
            if fk.table.lower() in keep and fk.ref_table.lower() in keep
        ]
        return DbSchema(db_id=self.db_id, tables=kept, foreign_keys=fks)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_create_sql(self, tables: Iterable[str] | None = None) -> str:
        """Render CREATE TABLE statements, including PRIMARY KEY / FOREIGN KEY lines.

        PK/FK constraints matter for text-to-SQL: they tell the model which
        columns to JOIN on, which is the most common failure mode on Spider.
        """
        schema = self.subset(tables) if tables is not None else self
        statements = []
        for t in schema.tables:
            lines = [f"  {quote_ident(c.name)} {c.type}" for c in t.columns]
            if t.primary_key:
                pk = ", ".join(quote_ident(c) for c in t.primary_key)
                lines.append(f"  PRIMARY KEY ({pk})")
            for fk in schema.foreign_keys:
                if fk.table == t.name:
                    lines.append(
                        f"  FOREIGN KEY ({quote_ident(fk.column)}) REFERENCES "
                        f"{quote_ident(fk.ref_table)}({quote_ident(fk.ref_column)})"
                    )
            statements.append(
                f"CREATE TABLE {quote_ident(t.name)} (\n" + ",\n".join(lines) + "\n);"
            )
        return "\n\n".join(statements)

    # ------------------------------------------------------------------
    # Constructors / serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_spider(cls, entry: dict) -> DbSchema:
        """Build from one entry of Spider's tables.json."""
        table_names = entry["table_names_original"]
        col_entries = entry["column_names_original"]  # [(table_idx, col_name), ...]
        col_types = entry["column_types"]

        tables = [Table(name=name, columns=[]) for name in table_names]
        col_ref: dict[int, tuple[str, str]] = {}  # column index -> (table, column)
        for c_idx, (t_idx, col) in enumerate(col_entries):
            if t_idx < 0 or col == "*":
                continue
            col_type = _SPIDER_TYPES.get(col_types[c_idx].lower(), col_types[c_idx].upper())
            tables[t_idx].columns.append(Column(name=col, type=col_type))
            col_ref[c_idx] = (table_names[t_idx], col)

        # Primary keys are column indices; newer dumps nest composite keys in lists
        for pk in entry.get("primary_keys", []):
            for c_idx in pk if isinstance(pk, list) else [pk]:
                if c_idx in col_ref:
                    t_name, col = col_ref[c_idx]
                    tables[table_names.index(t_name)].primary_key.append(col)

        fks = []
        for src_idx, dst_idx in entry.get("foreign_keys", []):
            if src_idx in col_ref and dst_idx in col_ref:
                (src_t, src_c), (dst_t, dst_c) = col_ref[src_idx], col_ref[dst_idx]
                fks.append(ForeignKey(src_t, src_c, dst_t, dst_c))
        return cls(db_id=entry["db_id"], tables=tables, foreign_keys=fks)

    @classmethod
    def from_sqlite(cls, db_path: str | Path, db_id: str | None = None) -> DbSchema:
        """Introspect an arbitrary SQLite database (bring-your-own-database)."""
        db_path = Path(db_path)
        uri = db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            names = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
                )
            ]
            tables, fks = [], []
            for name in names:
                info = conn.execute(f"PRAGMA table_info({quote_ident(name)})").fetchall()
                cols = [Column(name=r[1], type=(r[2] or "TEXT").upper()) for r in info]
                pk = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5] > 0]
                tables.append(Table(name=name, columns=cols, primary_key=pk))
                for r in conn.execute(f"PRAGMA foreign_key_list({quote_ident(name)})"):
                    ref_table, from_col, to_col = r[2], r[3], r[4]
                    if to_col is None:  # implicit reference to the parent's primary key
                        parent = conn.execute(
                            f"PRAGMA table_info({quote_ident(ref_table)})"
                        ).fetchall()
                        to_col = next((p[1] for p in parent if p[5] == 1), from_col)
                    fks.append(ForeignKey(name, from_col, ref_table, to_col))
        finally:
            conn.close()
        return cls(db_id=db_id or db_path.stem, tables=tables, foreign_keys=fks)

    def to_dict(self) -> dict:
        return {
            "db_id": self.db_id,
            "tables": [
                {
                    "name": t.name,
                    "columns": [[c.name, c.type] for c in t.columns],
                    "primary_key": t.primary_key,
                }
                for t in self.tables
            ],
            "foreign_keys": [
                [f.table, f.column, f.ref_table, f.ref_column] for f in self.foreign_keys
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> DbSchema:
        tables = [
            Table(
                name=t["name"],
                columns=[Column(name=n, type=ty) for n, ty in t["columns"]],
                primary_key=list(t.get("primary_key", [])),
            )
            for t in data["tables"]
        ]
        fks = [ForeignKey(*fk) for fk in data.get("foreign_keys", [])]
        return cls(db_id=data["db_id"], tables=tables, foreign_keys=fks)


class SchemaCatalog:
    """Registry of known databases: db_id -> DbSchema (+ SQLite file location)."""

    def __init__(self, schemas: dict[str, DbSchema], db_dirs: Iterable[str | Path] = ()):
        self._schemas = schemas
        self._db_dirs = [Path(d) for d in db_dirs]

    def __contains__(self, db_id: str) -> bool:
        return db_id in self._schemas

    def __len__(self) -> int:
        return len(self._schemas)

    def ids(self) -> list[str]:
        return sorted(self._schemas)

    def get(self, db_id: str) -> DbSchema:
        try:
            return self._schemas[db_id]
        except KeyError:
            raise KeyError(f"Unknown db_id '{db_id}'") from None

    def db_path(self, db_id: str) -> Path | None:
        """Spider layout: <db_dir>/<db_id>/<db_id>.sqlite."""
        for d in self._db_dirs:
            candidate = d / db_id / f"{db_id}.sqlite"
            if candidate.exists():
                return candidate
        return None

    def add(self, schema: DbSchema) -> None:
        self._schemas[schema.db_id] = schema

    @classmethod
    def from_json(cls, path: str | Path, db_dirs: Iterable[str | Path] = ()) -> SchemaCatalog:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls({k: DbSchema.from_dict(v) for k, v in raw.items()}, db_dirs)

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in sorted(self._schemas.items())}, f)
