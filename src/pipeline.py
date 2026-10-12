"""
Text-to-SQL pipeline — the single code path shared by the API and the evaluation harness.

  route db → cache → retrieve context → prompt → LLM (base or LoRA adapter)
           → extract SQL → validate (safety layer) → [repair] → guarded execution

Benchmark numbers are therefore measured on exactly the code that serves traffic.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import asdict, dataclass, field

from src.generation.postprocess import extract_sql
from src.generation.prompts import CANNOT_ANSWER, MODES, PROMPT_VERSION, build_prompt
from src.retrieval.context import ContextBuilder
from src.safety.executor import ExecutionError, QueryTimeoutError, SafeSQLiteExecutor
from src.safety.sql_validator import UNSAFE_CODES, SQLValidator, ValidationResult
from src.schema import SchemaCatalog
from src.serving import telemetry as m
from src.serving.cache import CacheError, QueryCache, make_cache_key
from src.serving.vllm_client import AdapterRegistry, LLMError, OpenAICompatibleClient

log = logging.getLogger(__name__)

# Final statuses (also the `status` label of text2sql_requests_total)
OK = "ok"
REJECTED = "rejected"
EXEC_ERROR = "exec_error"
CANNOT_ANSWER_STATUS = "cannot_answer"
LLM_ERROR = "llm_error"


class UnknownDatabaseError(KeyError):
    pass


@dataclass
class QueryRequest:
    question: str
    db_id: str | None = None
    mode: str = "rag"
    adapter: str | None = None
    execute: bool = True
    max_rows: int | None = 100  # None = fetch everything (evaluation)
    use_cache: bool = True


@dataclass
class QueryResult:
    status: str
    question: str
    db_id: str
    mode: str
    model: str
    adapter: str | None = None
    sql: str | None = None
    raw_output: str | None = None
    routed: bool = False
    cached: bool = False
    repaired: bool = False
    repair_attempts: int = 0
    violations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    columns: list[str] | None = None
    rows: list[list] | None = None
    row_count: int | None = None
    truncated: bool = False
    error: str | None = None
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    timings: dict = field(default_factory=dict)
    context_tables: list[str] = field(default_factory=list)
    num_examples: int = 0
    value_hints: list[str] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.rows is not None:
            data["rows"] = [[_jsonable(v) for v in row] for row in self.rows]
        return data


def _jsonable(value):
    if isinstance(value, bytes):
        return "base64:" + base64.b64encode(value).decode("ascii")
    return value


def _repair_messages(previous_output: str, problem: str) -> list[dict]:
    return [
        {"role": "assistant", "content": previous_output},
        {
            "role": "user",
            "content": (
                f"The SQL query above is invalid: {problem}\n"
                "Using only the tables and columns in the schema, write a corrected SQLite "
                "query. Output ONLY the SQL."
            ),
        },
    ]


class Text2SQLPipeline:
    def __init__(
        self,
        *,
        catalog: SchemaCatalog,
        context_builder: ContextBuilder,
        llm: OpenAICompatibleClient,
        registry: AdapterRegistry,
        validator: SQLValidator | None = None,
        executor: SafeSQLiteExecutor | None = None,
        cache: QueryCache | None = None,
        cache_ttl_s: int = 3600,
        max_tokens: int = 256,
        max_repair_attempts: int = 1,
    ):
        self.catalog = catalog
        self.context_builder = context_builder
        self.llm = llm
        self.registry = registry
        self.validator = validator or SQLValidator()
        self.executor = executor or SafeSQLiteExecutor()
        self.cache = cache
        self.cache_ttl_s = cache_ttl_s
        self.max_tokens = max_tokens
        self.max_repair_attempts = max_repair_attempts

    async def run(self, req: QueryRequest) -> QueryResult:
        """Run one request end to end. Raises UnknownDatabaseError / AdapterNotFoundError
        / ValueError for bad requests; every other outcome is reported in `status`."""
        if req.mode not in MODES:
            raise ValueError(f"Unknown mode '{req.mode}'")
        start = time.perf_counter()
        m.INFLIGHT.inc()
        try:
            result = await self._run(req)
        finally:
            m.INFLIGHT.dec()
        total = time.perf_counter() - start
        result.timings["total_ms"] = round(total * 1000, 2)
        m.REQUESTS.labels(req.mode, result.status).inc()
        m.REQUEST_LATENCY.labels(req.mode).observe(total)
        return result

    async def _run(self, req: QueryRequest) -> QueryResult:
        db_id, routed = req.db_id, False
        if db_id is None:
            db_id, routed = self.context_builder.schema_retriever.route(req.question), True
        if db_id not in self.catalog:
            raise UnknownDatabaseError(db_id)
        schema = self.catalog.get(db_id)
        model, adapter = self.registry.resolve(req.mode, req.adapter)

        result = QueryResult(
            status=OK, question=req.question, db_id=db_id, mode=req.mode,
            model=model, adapter=adapter, routed=routed,
        )  # fmt: skip
        timings = result.timings

        # ---- cache: generation is the expensive, deterministic part --------------------
        key = make_cache_key(
            question=req.question, db_id=db_id, mode=req.mode, model=model,
            prompt_version=PROMPT_VERSION,
        )  # fmt: skip
        cached = await self._cache_get(key) if req.use_cache else None
        if cached is not None:
            result.cached = True
            for name in ("sql", "raw_output", "repaired", "repair_attempts", "usage",
                         "context_tables", "num_examples", "value_hints"):  # fmt: skip
                setattr(result, name, cached.get(name, getattr(result, name)))
            if result.sql == CANNOT_ANSWER:
                result.status = CANNOT_ANSWER_STATUS
                return result
            validation = self._validate(result.sql, schema, timings)
            if not validation.ok:  # e.g. validator rules tightened since it was cached
                return self._reject(result, validation)
            await self._execute(result, validation, req, timings)
            return result

        # ---- retrieval -----------------------------------------------------------------
        t = time.perf_counter()
        ctx = self.context_builder.build(req.mode, req.question, db_id)
        self._observe(timings, "retrieval", t)
        result.context_tables, result.num_examples = ctx.tables, len(ctx.examples)
        result.value_hints = ctx.value_hints
        messages = build_prompt(
            req.question, ctx.schema_sql, req.mode, ctx.examples, ctx.value_hints
        )

        # ---- generate → validate → (repair) → execute ------------------------------------
        while True:
            t = time.perf_counter()
            try:
                gen = await self.llm.chat(messages, model=model, max_tokens=self.max_tokens)
            except LLMError as e:
                self._observe(timings, "generation", t)
                m.LLM_ERRORS.labels(e.kind).inc()
                result.status, result.error = LLM_ERROR, str(e)
                return result
            self._observe(timings, "generation", t)
            m.ADAPTER_REQUESTS.labels(adapter or "base").inc()
            m.LLM_TOKENS.labels(req.mode, "prompt").inc(gen.prompt_tokens)
            m.LLM_TOKENS.labels(req.mode, "completion").inc(gen.completion_tokens)
            result.usage["prompt_tokens"] += gen.prompt_tokens
            result.usage["completion_tokens"] += gen.completion_tokens
            result.raw_output = gen.text
            result.sql = extract_sql(gen.text)

            if result.sql == CANNOT_ANSWER:
                result.status = CANNOT_ANSWER_STATUS
                break

            validation = self._validate(result.sql, schema, timings)
            if not validation.ok:
                unsafe = bool(UNSAFE_CODES & set(validation.codes))
                if not unsafe and result.repair_attempts < self.max_repair_attempts:
                    messages = messages + _repair_messages(gen.text, validation.feedback())
                    result.repair_attempts += 1
                    continue
                self._reject(result, validation)
                break

            result.violations, result.warnings = [], validation.warnings
            retryable = await self._execute(result, validation, req, timings)
            if retryable and result.repair_attempts < self.max_repair_attempts:
                messages = messages + _repair_messages(gen.text, result.error)
                result.repair_attempts += 1
                result.status, result.error = OK, None
                continue
            break

        if result.repair_attempts:
            result.repaired = result.status == OK
            m.SQL_REPAIRS.labels("success" if result.repaired else "failure").inc()
        if result.status in (OK, CANNOT_ANSWER_STATUS) and req.use_cache:
            await self._cache_set(key, result)
        return result

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _validate(self, sql: str, schema, timings: dict) -> ValidationResult:
        t = time.perf_counter()
        validation = self.validator.validate(sql, schema)
        self._observe(timings, "validation", t)
        for code in set(validation.codes):
            m.VALIDATION_FAILURES.labels(code).inc()
        return validation

    @staticmethod
    def _reject(result: QueryResult, validation: ValidationResult) -> QueryResult:
        result.status = REJECTED
        result.violations = [v.to_dict() for v in validation.violations]
        result.warnings = validation.warnings
        result.error = validation.feedback()
        return result

    async def _execute(
        self, result: QueryResult, validation: ValidationResult, req: QueryRequest, timings: dict
    ) -> bool:
        """Run the validated statement. Returns True when a repair attempt could help."""
        result.sql = validation.safe_sql
        if not req.execute:
            return False
        db_path = self.catalog.db_path(result.db_id)
        if db_path is None:
            result.status, result.error = EXEC_ERROR, f"No database file for '{result.db_id}'"
            return False
        t = time.perf_counter()
        retryable = False
        try:
            out = await asyncio.to_thread(self.executor.execute, db_path, result.sql, req.max_rows)
        except QueryTimeoutError as e:  # a rewrite won't make an expensive query cheap
            result.status, result.error = EXEC_ERROR, str(e)
        except ExecutionError as e:
            result.status, result.error = EXEC_ERROR, f"Database error: {e}"
            retryable = True
        else:
            result.status = OK
            result.columns, result.rows = out.columns, [list(r) for r in out.rows]
            result.row_count, result.truncated = out.row_count, out.truncated
        self._observe(timings, "execution", t)
        return retryable

    async def _cache_get(self, key: str) -> dict | None:
        if self.cache is None:
            return None
        try:
            value = await self.cache.get(key)
        except CacheError as e:
            log.warning("cache read failed: %s", e)
            m.CACHE_REQUESTS.labels("error").inc()
            return None
        m.CACHE_REQUESTS.labels("hit" if value is not None else "miss").inc()
        return value

    async def _cache_set(self, key: str, result: QueryResult) -> None:
        if self.cache is None:
            return
        payload = {
            "sql": result.sql, "raw_output": result.raw_output, "repaired": result.repaired,
            "repair_attempts": result.repair_attempts, "usage": result.usage,
            "context_tables": result.context_tables, "num_examples": result.num_examples,
            "value_hints": result.value_hints,
        }  # fmt: skip
        try:
            await self.cache.set(key, payload, self.cache_ttl_s)
        except CacheError as e:
            log.warning("cache write failed: %s", e)
            m.CACHE_REQUESTS.labels("error").inc()

    @staticmethod
    def _observe(timings: dict, stage: str, started: float) -> None:
        elapsed = time.perf_counter() - started
        m.STAGE_LATENCY.labels(stage).observe(elapsed)
        timings[f"{stage}_ms"] = round(timings.get(f"{stage}_ms", 0.0) + elapsed * 1000, 2)
