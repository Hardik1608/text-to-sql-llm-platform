"""
Prometheus metrics for the Text-to-SQL service.

Metric names and labels are a contract with infra/alerts.yml and the Grafana dashboard
(monitoring/grafana/dashboards/text2sql-overview.json) — change them together.
Label values are bounded enums (mode, status, stage, reason, ...) to keep cardinality low.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
STAGE_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32)

REQUESTS = Counter(
    "text2sql_requests_total",
    "Text-to-SQL requests by mode and outcome",
    ["mode", "status"],  # status: ok|rejected|exec_error|cannot_answer|llm_error
)
REQUEST_LATENCY = Histogram(
    "text2sql_request_latency_seconds",
    "End-to-end request latency",
    ["mode"],
    buckets=LATENCY_BUCKETS,
)
STAGE_LATENCY = Histogram(
    "text2sql_stage_latency_seconds",
    "Latency per pipeline stage",
    ["stage"],  # retrieval|generation|validation|execution
    buckets=STAGE_BUCKETS,
)
LLM_TOKENS = Counter(
    "text2sql_llm_tokens_total",
    "Tokens processed by the LLM",
    ["mode", "kind"],  # kind: prompt|completion
)
LLM_ERRORS = Counter(
    "text2sql_llm_errors_total",
    "LLM backend failures",
    ["kind"],  # timeout|connection|http_error|bad_response
)
CACHE_REQUESTS = Counter(
    "text2sql_cache_requests_total",
    "Query cache lookups",
    ["result"],  # hit|miss|error
)
VALIDATION_FAILURES = Counter(
    "text2sql_validation_failures_total",
    "Generated SQL rejected by the safety layer",
    ["reason"],
)
SQL_REPAIRS = Counter(
    "text2sql_sql_repairs_total",
    "Self-repair attempts after validation/execution failures",
    ["outcome"],  # success|failure
)
ADAPTER_REQUESTS = Counter(
    "text2sql_adapter_requests_total",
    "Generations per LoRA adapter ('base' = no adapter)",
    ["adapter"],
)
INFLIGHT = Gauge("text2sql_inflight_requests", "Requests currently being processed")
BUILD_INFO = Gauge("text2sql_build_info", "Build information", ["version", "base_model"])


def init_metrics(version: str, base_model: str, modes: tuple[str, ...]) -> None:
    """Create every labelled series at 0 so dashboards show 0 instead of "No data"."""
    BUILD_INFO.labels(version, base_model).set(1)
    for mode in modes:
        REQUEST_LATENCY.labels(mode)
        for status in ("ok", "rejected", "exec_error", "cannot_answer", "llm_error"):
            REQUESTS.labels(mode, status)
        for kind in ("prompt", "completion"):
            LLM_TOKENS.labels(mode, kind)
    for stage in ("retrieval", "generation", "validation", "execution"):
        STAGE_LATENCY.labels(stage)
    for kind in ("timeout", "connection", "http_error", "bad_response"):
        LLM_ERRORS.labels(kind)
    for result in ("hit", "miss", "error"):
        CACHE_REQUESTS.labels(result)
    for outcome in ("success", "failure"):
        SQL_REPAIRS.labels(outcome)
    for reason in (
        "empty", "too_long", "parse_error", "multiple_statements", "not_read_only",
        "forbidden_function", "unknown_table", "unknown_column",
    ):  # fmt: skip
        VALIDATION_FAILURES.labels(reason)
