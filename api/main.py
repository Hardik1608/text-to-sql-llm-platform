"""
FastAPI gateway for the Text-to-SQL platform.

    GET  /                      web UI (api/static/): ask a question, see SQL + rows
    GET  /info                  service info (version, modes, base model)
    POST /query                 natural language → validated SQL → rows
    POST /compare               one question through all 4 approaches, graded when gold is known
    GET  /health, /ready        liveness / readiness (LLM backend reachable)
    GET  /metrics               Prometheus metrics
    GET  /databases[/{db_id}]   known databases and their schemas
    GET  /adapters              registered LoRA adapters
    POST /adapters/load|unload  hot-swap adapters on vLLM (requires X-Admin-Token)

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, field_validator

from src.config import Settings, get_settings
from src.factory import build_pipeline
from src.generation.prompts import MODES, PROMPT_VERSION
from src.pipeline import LLM_ERROR, QueryRequest, Text2SQLPipeline, UnknownDatabaseError
from src.serving import telemetry as m
from src.serving.compare import BenchmarkIndex, compare_modes
from src.serving.vllm_client import AdapterNotFoundError, LLMError

__version__ = "0.2.0"
log = logging.getLogger("text2sql.api")
STATIC_DIR = Path(__file__).parent / "static"  # the single-page UI served at /


def clean_question(text: str) -> str:
    """Drop control characters and collapse whitespace; reject what is left if it is empty."""
    cleaned = " ".join("".join(ch for ch in text if ch.isprintable() or ch.isspace()).split())
    if len(cleaned) < 3:
        raise ValueError("question is empty")
    return cleaned


class QueryBody(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=1000, examples=["How many singers are there?"]
    )
    db_id: str | None = Field(None, max_length=128, examples=["concert_singer"])
    mode: Literal["prompt", "rag", "qlora", "qlora_rag"] = "rag"
    adapter: str | None = Field(
        None, max_length=128, description="LoRA adapter version, e.g. spider-v1"
    )
    execute: bool = True
    max_rows: int = Field(100, ge=1, le=1000)
    use_cache: bool = True

    @field_validator("question")
    @classmethod
    def _clean_question(cls, v: str) -> str:
        return clean_question(v)

    @field_validator("db_id", "adapter")
    @classmethod
    def _strip_names(cls, v: str | None) -> str | None:
        # A copy-pasted trailing space must not turn into "Unknown database 'x '"
        return (v.strip() or None) if v is not None else None


class CompareBody(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    db_id: str = Field(..., max_length=128)
    use_cache: bool = False  # default off: the comparison should show real generation latency

    @field_validator("question")
    @classmethod
    def _clean_question(cls, v: str) -> str:
        return clean_question(v)

    @field_validator("db_id")
    @classmethod
    def _strip_db(cls, v: str) -> str:
        return v.strip()


class AdapterLoadBody(BaseModel):
    name: str = Field(..., pattern=r"^[A-Za-z0-9._-]{1,64}$")
    path: str = Field(..., min_length=1, max_length=512)


class AdapterUnloadBody(BaseModel):
    name: str = Field(..., pattern=r"^[A-Za-z0-9._-]{1,64}$")


def create_app(
    settings: Settings | None = None, pipeline: Text2SQLPipeline | None = None
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or build_pipeline(settings)
        app.state.benchmark = BenchmarkIndex.load(settings.processed_dir)
        m.init_metrics(__version__, settings.base_model, MODES)
        catalog = app.state.pipeline.catalog
        if len(catalog):  # warm sqlglot's lazily-imported optimizer (~0.2 s) before traffic
            app.state.pipeline.validator.validate("SELECT 1", catalog.get(catalog.ids()[0]))
        log.info(
            "ready: %d databases, base model %s, adapters %s",
            len(app.state.pipeline.catalog),
            settings.base_model,
            [a.name for a in app.state.pipeline.registry.list()] or "none",
        )
        yield
        await app.state.pipeline.llm.aclose()
        if app.state.pipeline.cache is not None:
            await app.state.pipeline.cache.aclose()

    app = FastAPI(
        title="Text-to-SQL LLM Platform",
        version=__version__,
        description="Prompting vs schema RAG vs QLoRA fine-tuning, served with a SQL safety layer.",
        lifespan=lifespan,
    )

    def _pipeline(request: Request) -> Text2SQLPipeline:
        return request.app.state.pipeline

    def _require_admin(token: str | None) -> None:
        if not settings.admin_token:
            raise HTTPException(403, "Admin endpoints are disabled (set T2S_ADMIN_TOKEN)")
        if not token or not secrets.compare_digest(token, settings.admin_token):
            raise HTTPException(401, "Invalid admin token")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def ui() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/info")
    async def info() -> dict:
        return {
            "service": "text-to-sql-llm-platform",
            "version": __version__,
            "modes": list(MODES),
            "base_model": settings.base_model,
            "prompt_version": PROMPT_VERSION,
        }

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        pipeline = _pipeline(request)
        llm_ok = await pipeline.llm.health()
        cache_ok = await pipeline.cache.ping() if pipeline.cache is not None else None
        body = {"llm": llm_ok, "cache": cache_ok, "databases": len(pipeline.catalog)}
        return JSONResponse(body, status_code=200 if llm_ok else 503)

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/query")
    async def query(body: QueryBody, request: Request) -> JSONResponse:
        req = QueryRequest(
            question=body.question,
            db_id=body.db_id,
            mode=body.mode,
            adapter=body.adapter,
            execute=body.execute,
            max_rows=body.max_rows,
            use_cache=body.use_cache,
        )
        try:
            result = await _pipeline(request).run(req)
        except UnknownDatabaseError as e:
            raise HTTPException(404, f"Unknown database '{e.args[0]}'") from None
        except AdapterNotFoundError as e:
            raise HTTPException(400, str(e.args[0])) from None
        log.info(
            "query mode=%s db=%s status=%s cached=%s repaired=%s total_ms=%s",
            result.mode, result.db_id, result.status, result.cached, result.repaired,
            result.timings.get("total_ms"),
        )  # fmt: skip
        status_code = 503 if result.status == LLM_ERROR else 200
        return JSONResponse(result.to_dict(), status_code=status_code)

    @app.post("/compare")
    async def compare(body: CompareBody, request: Request) -> dict:
        """Run one question through all four approaches; grade them when the answer is known."""
        pipeline = _pipeline(request)
        if body.db_id not in pipeline.catalog:
            raise HTTPException(404, f"Unknown database '{body.db_id}'")
        return await compare_modes(
            pipeline, request.app.state.benchmark, body.question, body.db_id,
            use_cache=body.use_cache,
        )  # fmt: skip

    @app.get("/databases/{db_id}/questions")
    async def benchmark_questions(db_id: str, request: Request, limit: int = 8) -> dict:
        """Held-out Spider questions for a database (their correct answer is known)."""
        db_id = db_id.strip()
        if db_id not in _pipeline(request).catalog:
            raise HTTPException(404, f"Unknown database '{db_id}'")
        items = request.app.state.benchmark.questions(db_id, max(1, min(limit, 50)))
        return {
            "db_id": db_id,
            "questions": [{"question": q.question, "split": q.split} for q in items],
        }

    @app.get("/databases")
    async def databases(request: Request) -> dict:
        catalog = _pipeline(request).catalog
        return {"databases": catalog.ids(), "count": len(catalog)}

    @app.get("/databases/{db_id}")
    async def database_schema(db_id: str, request: Request) -> PlainTextResponse:
        catalog = _pipeline(request).catalog
        db_id = db_id.strip()
        if db_id not in catalog:
            raise HTTPException(404, f"Unknown database '{db_id}'")
        return PlainTextResponse(catalog.get(db_id).to_create_sql())

    @app.get("/adapters")
    async def adapters(request: Request) -> dict:
        registry = _pipeline(request).registry
        return {
            "base_model": registry.base_model,
            "default": registry.default,
            "adapters": [a.to_dict() for a in registry.list()],
        }

    @app.post("/adapters/load")
    async def load_adapter(
        body: AdapterLoadBody, request: Request, x_admin_token: str | None = Header(None)
    ) -> dict:
        _require_admin(x_admin_token)
        pipeline = _pipeline(request)
        try:
            await pipeline.llm.load_lora_adapter(body.name, body.path)
        except LLMError as e:
            raise HTTPException(502, f"vLLM refused to load the adapter: {e}") from None
        pipeline.registry.register(body.name, body.path)
        return {"loaded": body.name, "default": pipeline.registry.default}

    @app.post("/adapters/unload")
    async def unload_adapter(
        body: AdapterUnloadBody, request: Request, x_admin_token: str | None = Header(None)
    ) -> dict:
        _require_admin(x_admin_token)
        pipeline = _pipeline(request)
        if body.name not in pipeline.registry:
            raise HTTPException(404, f"Unknown adapter '{body.name}'")
        try:
            await pipeline.llm.unload_lora_adapter(body.name)
        except LLMError as e:
            raise HTTPException(502, f"vLLM refused to unload the adapter: {e}") from None
        pipeline.registry.unregister(body.name)
        return {"unloaded": body.name, "default": pipeline.registry.default}

    return app


app = create_app()
