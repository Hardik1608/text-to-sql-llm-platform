"""
Few-shot example retrieval (the "R" that helps most in Text-to-SQL RAG).

Retrieves training (question, SQL) pairs whose questions are *structurally* similar to the
incoming question, and uses them as dynamic few-shot demonstrations.

Similarity is computed on schema-masked questions: tokens that name tables/columns of the
question's own database are removed (as are numbers and quoted values), leaving the
structural skeleton ("how many ... more than ... each") — the idea behind DAIL-SQL's masked
question similarity. Topical overlap is useless here: in Spider, evaluation databases never
appear in the training set.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.retrieval.bm25 import BM25
from src.retrieval.text import split_identifier, tokenize
from src.schema import DbSchema, SchemaCatalog

_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_SQL_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"|\b\d+(?:\.\d+)?\b")


@dataclass(frozen=True)
class Example:
    question: str
    query: str
    db_id: str

    def to_dict(self) -> dict[str, str]:
        return {"question": self.question, "query": self.query, "db_id": self.db_id}


def _schema_vocab(schema: DbSchema | None) -> set[str]:
    if schema is None:
        return set()
    vocab = set(split_identifier(schema.db_id))
    for table in schema.tables:
        vocab.update(split_identifier(table.name))
        for col in table.column_names:
            vocab.update(split_identifier(col))
    return vocab


def question_skeleton(question: str, schema: DbSchema | None = None) -> list[str]:
    """Structural tokens of a question: values and schema mentions masked out."""
    text = _NUMBER.sub(" ", _QUOTED.sub(" ", question))
    vocab = _schema_vocab(schema)
    return [t for t in tokenize(text, remove_stopwords=False) if t not in vocab]


def _sql_template(sql: str) -> str:
    return " ".join(_SQL_LITERAL.sub("?", sql.lower()).split())


class ExampleRetriever:
    def __init__(self, examples: list[Example], catalog: SchemaCatalog | None = None):
        if not examples:
            raise ValueError("ExampleRetriever needs at least one example")
        self.examples = examples
        self.catalog = catalog
        corpus = [question_skeleton(ex.question, self._schema(ex.db_id)) for ex in examples]
        self._bm25 = BM25([doc or ["<empty>"] for doc in corpus])

    @classmethod
    def from_records(
        cls, rows: list[dict], catalog: SchemaCatalog | None = None
    ) -> ExampleRetriever:
        return cls([Example(r["question"], r["query"], r["db_id"]) for r in rows], catalog)

    @classmethod
    def from_jsonl(cls, path: str | Path, catalog: SchemaCatalog | None = None) -> ExampleRetriever:
        with open(path, encoding="utf-8") as f:
            return cls.from_records([json.loads(line) for line in f if line.strip()], catalog)

    def retrieve(
        self,
        question: str,
        k: int = 3,
        *,
        db_id: str | None = None,
        exclude_db_id: str | None = None,
        exclude_question: str | None = None,
    ) -> list[Example]:
        """Top-k structurally similar examples with distinct SQL templates.

        `exclude_db_id` drops examples from the question's own database. Use it when
        building fine-tuning data so training conditions match evaluation, where the
        retrieved examples always come from *other* databases.
        """
        if k <= 0:
            return []
        tokens = question_skeleton(question, self._schema(db_id)) or tokenize(question, False)
        scores = self._bm25.get_scores(tokens)
        picked: list[Example] = []
        seen_templates: set[str] = set()
        for idx in np.argsort(-scores, kind="stable"):
            ex = self.examples[int(idx)]
            if exclude_db_id is not None and ex.db_id == exclude_db_id:
                continue
            if exclude_question is not None and ex.question == exclude_question:
                continue
            template = _sql_template(ex.query)
            if template in seen_templates:
                continue
            seen_templates.add(template)
            picked.append(ex)
            if len(picked) == k:
                break
        return picked

    def _schema(self, db_id: str | None) -> DbSchema | None:
        if db_id is None or self.catalog is None or db_id not in self.catalog:
            return None
        return self.catalog.get(db_id)
