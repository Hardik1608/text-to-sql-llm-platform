"""
Context builder — the single place that decides what each mode puts in the prompt.

Used by the API, the evaluation harness and fine-tuning data preparation, so training-time
and inference-time prompts can never drift apart.

  mode        schema            examples                  value hints
  ---------   ---------------   -----------------------   -----------
  prompt      full              static (prompt template)  -
  qlora       full              -                         -
  rag         BM25-pruned       retrieved (BM25)          retrieved
  qlora_rag   BM25-pruned       retrieved (BM25)          retrieved
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from src.generation.prompts import RAG_MODES
from src.retrieval.example_retriever import ExampleRetriever
from src.retrieval.schema_retriever import SchemaRetriever
from src.retrieval.value_index import ValueIndex
from src.schema import SchemaCatalog


@dataclass
class Context:
    schema_sql: str
    tables: list[str]
    examples: list[dict] = field(default_factory=list)
    value_hints: list[str] = field(default_factory=list)


class ContextBuilder:
    def __init__(
        self,
        catalog: SchemaCatalog,
        schema_retriever: SchemaRetriever,
        example_retriever: ExampleRetriever | None = None,
        value_index: ValueIndex | None = None,
        num_examples: int = 3,
        max_tables: int | None = 6,
    ):
        self.catalog = catalog
        self.schema_retriever = schema_retriever
        self.example_retriever = example_retriever
        self.value_index = value_index
        self.num_examples = num_examples
        self.max_tables = max_tables

    def build(
        self,
        mode: str,
        question: str,
        db_id: str,
        *,
        exclude_example_db: str | None = None,
        must_include_tables: Iterable[str] = (),
    ) -> Context:
        schema = self.catalog.get(db_id)
        if mode not in RAG_MODES:
            return Context(schema_sql=schema.to_create_sql(), tables=schema.table_names)

        selected = set(self.schema_retriever.select_tables(question, db_id, self.max_tables))
        selected |= {t for t in must_include_tables if schema.table(t) is not None}
        tables = [t for t in schema.table_names if t in selected]

        examples = []
        if self.example_retriever is not None and self.num_examples > 0:
            examples = [
                {"question": ex.question, "query": ex.query}
                for ex in self.example_retriever.retrieve(
                    question,
                    k=self.num_examples,
                    db_id=db_id,
                    exclude_db_id=exclude_example_db,
                    exclude_question=question,
                )
            ]

        hints = []
        if self.value_index is not None:
            hints = [h.render() for h in self.value_index.match(question, db_id, tables)]

        return Context(
            schema_sql=schema.to_create_sql(tables),
            tables=tables,
            examples=examples,
            value_hints=hints,
        )
