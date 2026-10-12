"""Wire the pipeline's components from Settings (shared by the API and the evaluator)."""

from __future__ import annotations

import logging

from src.config import Settings
from src.pipeline import Text2SQLPipeline
from src.retrieval.context import ContextBuilder
from src.retrieval.example_retriever import ExampleRetriever
from src.retrieval.schema_retriever import SchemaRetriever
from src.retrieval.value_index import ValueIndex
from src.safety.executor import SafeSQLiteExecutor
from src.safety.sql_validator import SQLValidator
from src.schema import SchemaCatalog
from src.serving.cache import InMemoryCache, QueryCache, RedisCache
from src.serving.vllm_client import AdapterRegistry, OpenAICompatibleClient, parse_adapters

log = logging.getLogger(__name__)


def build_catalog(settings: Settings) -> SchemaCatalog:
    schemas_file = settings.processed_dir / "schemas.json"
    if not schemas_file.exists():
        raise FileNotFoundError(
            f"{schemas_file} not found — run `python -m src.ingestion.download_spider` first"
        )
    return SchemaCatalog.from_json(schemas_file, settings.db_dirs)


def build_cache(settings: Settings) -> QueryCache | None:
    if not settings.cache_enabled:
        return None
    if settings.redis_url.startswith("memory://"):
        return InMemoryCache()
    return RedisCache(settings.redis_url)


def build_pipeline(
    settings: Settings,
    *,
    llm: OpenAICompatibleClient | None = None,
    cache: QueryCache | None | bool = True,
) -> Text2SQLPipeline:
    """`cache=True` builds the configured cache; pass None to disable or an instance to inject."""
    catalog = build_catalog(settings)
    schema_retriever = SchemaRetriever.from_catalog(catalog)

    examples_file = settings.processed_dir / "train.jsonl"
    example_retriever = None
    if settings.rag_num_examples > 0 and examples_file.exists():
        example_retriever = ExampleRetriever.from_jsonl(examples_file, catalog)
    elif settings.rag_num_examples > 0:
        log.warning("%s not found: RAG modes will run without retrieved examples", examples_file)

    context_builder = ContextBuilder(
        catalog=catalog,
        schema_retriever=schema_retriever,
        example_retriever=example_retriever,
        value_index=ValueIndex(catalog) if settings.rag_value_hints else None,
        num_examples=settings.rag_num_examples,
        max_tables=settings.rag_max_tables,
    )
    registry = AdapterRegistry(
        base_model=settings.base_model,
        adapters=parse_adapters(settings.adapters),
        default=settings.default_adapter,
    )
    llm = llm or OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )
    return Text2SQLPipeline(
        catalog=catalog,
        context_builder=context_builder,
        llm=llm,
        registry=registry,
        validator=SQLValidator(check_columns=settings.check_columns),
        executor=SafeSQLiteExecutor(timeout_s=settings.query_timeout_s, max_rows=settings.max_rows),
        cache=build_cache(settings) if cache is True else (cache or None),
        cache_ttl_s=settings.cache_ttl_seconds,
        max_tokens=settings.llm_max_tokens,
        max_repair_attempts=settings.max_repair_attempts,
    )
