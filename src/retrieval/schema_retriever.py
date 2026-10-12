"""
Schema Retriever — BM25-based retrieval over database schemas.

Two granularities:
  1. Database routing  : which database does a question target? (used when the caller
                         does not pass a db_id; each database is one BM25 document)
  2. Table pruning     : which tables of the target database are relevant? (schema RAG —
                         a shorter, more focused prompt; essential for large warehouses
                         where the full schema does not fit the context window)

For Text-to-SQL, BM25 over identifier tokens is a strong, cheap baseline: schemas are
structured, and questions usually mention table/column names (or their plurals) directly.
Pruned schemas are expanded with foreign-key neighbours so join paths stay intact.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.retrieval.bm25 import BM25
from src.retrieval.text import split_identifier, tokenize
from src.schema import DbSchema, SchemaCatalog


def _db_tokens(schema: DbSchema) -> list[str]:
    tokens = split_identifier(schema.db_id)
    for table in schema.tables:
        tokens += split_identifier(table.name) * 2  # table names weigh more than columns
        for col in table.column_names:
            tokens += split_identifier(col)
    return tokens


def _table_tokens(schema: DbSchema, table_name: str) -> list[str]:
    table = schema.table(table_name)
    tokens = split_identifier(table.name) * 3
    for col in table.column_names:
        tokens += split_identifier(col)
    return tokens


class SchemaRetriever:
    """
    BM25-based schema retriever.

    At index time: each DB schema is a "document" (tokenized table + column names).
    At query time: returns the most relevant databases, or the most relevant tables
    of one database.
    """

    def __init__(
        self, processed_dir: str | Path = "data/processed", catalog: SchemaCatalog | None = None
    ):
        self.processed_dir = Path(processed_dir)
        self.catalog = catalog
        self._db_schemas: dict[str, str] = {}  # {db_id: schema_sql}
        self._db_ids: list[str] = []
        self._bm25: BM25 | None = None
        self._table_index: dict[str, tuple[list[str], BM25]] = {}

    @classmethod
    def from_catalog(cls, catalog: SchemaCatalog) -> SchemaRetriever:
        retriever = cls(catalog=catalog)
        retriever.build_index()
        return retriever

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(self, split: str | None = None) -> None:
        """Build the database-level BM25 index.

        Indexes every database in the catalog (data/processed/schemas.json). Passing a
        `split` restricts the index to the databases that appear in that split's JSONL.
        """
        if self.catalog is None:
            self.catalog = SchemaCatalog.from_json(self.processed_dir / "schemas.json")
        db_ids = self.catalog.ids()
        if split is not None:
            with open(self.processed_dir / f"{split}.jsonl", encoding="utf-8") as f:
                in_split = {json.loads(line)["db_id"] for line in f}
            db_ids = [d for d in db_ids if d in in_split]

        self._db_ids = db_ids
        self._db_schemas = {d: self.catalog.get(d).to_create_sql() for d in db_ids}
        self._bm25 = BM25([_db_tokens(self.catalog.get(d)) for d in db_ids])
        self._table_index.clear()

    def load_schema(self, db_id: str) -> str:
        """Return the full schema SQL for a known db_id."""
        if db_id not in self._db_schemas:
            raise KeyError(f"db_id '{db_id}' not in index. Did you build_index()?")
        return self._db_schemas[db_id]

    # ------------------------------------------------------------------
    # Database routing
    # ------------------------------------------------------------------

    def retrieve(self, question: str, db_id: str | None = None, top_k: int = 3) -> list[dict]:
        """
        Retrieve the top-k most relevant database schemas.

        If db_id is provided: return only that DB's schema (exact match — used when the
        system already knows which DB to query). Otherwise rank all schemas by BM25.
        """
        if db_id is not None:
            return [{"db_id": db_id, "schema_sql": self.load_schema(db_id), "score": 1.0}]
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        scores = self._bm25.get_scores(tokenize(question))
        ranked = sorted(zip(scores, self._db_ids, strict=True), key=lambda x: x[0], reverse=True)
        return [
            {"db_id": did, "schema_sql": self._db_schemas[did], "score": float(score)}
            for score, did in ranked[:top_k]
        ]

    def route(self, question: str) -> str:
        """Best-matching db_id for a question that did not specify one."""
        return self.retrieve(question, top_k=1)[0]["db_id"]

    # ------------------------------------------------------------------
    # Table pruning (schema RAG within one database)
    # ------------------------------------------------------------------

    def rank_tables(self, question: str, db_id: str) -> list[tuple[str, float]]:
        """All tables of `db_id`, scored by BM25 relevance to the question (desc)."""
        names, bm25 = self._tables_for(db_id)
        scores = bm25.get_scores(tokenize(question))
        order = sorted(range(len(names)), key=lambda i: (-scores[i], i))
        return [(names[i], float(scores[i])) for i in order]

    def select_tables(
        self,
        question: str,
        db_id: str,
        max_tables: int | None = None,
        expand_foreign_keys: bool = True,
    ) -> list[str]:
        """Tables to put in the prompt, in schema order.

        Keeps the top `max_tables` tables by BM25 (all tables if the database is small or no
        table matches at all), then adds foreign-key neighbours of the selection so the model
        can still join through bridge tables.
        """
        schema = self.catalog.get(db_id)
        if max_tables is None or len(schema.tables) <= max_tables:
            return schema.table_names
        ranked = self.rank_tables(question, db_id)
        if ranked[0][1] <= 0:
            return schema.table_names  # no lexical signal -> don't guess
        selected = {name for name, score in ranked[:max_tables] if score > 0}
        if expand_foreign_keys:
            for name in list(selected):
                selected |= schema.neighbors(name)
        return [t for t in schema.table_names if t in selected]

    def _tables_for(self, db_id: str) -> tuple[list[str], BM25]:
        if db_id not in self._table_index:
            schema = self.catalog.get(db_id)
            names = schema.table_names
            self._table_index[db_id] = (names, BM25([_table_tokens(schema, n) for n in names]))
        return self._table_index[db_id]

    def format_context(self, results: list[dict]) -> str:
        """Format retrieved schemas as a prompt-ready context string."""
        parts = []
        for r in results:
            parts.append(f"-- Database: {r['db_id']}\n{r['schema_sql']}")
        return "\n\n".join(parts)


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    retriever = SchemaRetriever()
    retriever.build_index()
    question = "How many singers are there?"
    print("routed to:", retriever.route(question))
    print(retriever.rank_tables(question, "concert_singer"))
    print(retriever.format_context(retriever.retrieve(question, db_id="concert_singer")))
