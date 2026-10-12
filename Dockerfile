# syntax=docker/dockerfile:1
# =============================================================================
# Text-to-SQL API image (FastAPI gateway).
#
# CPU-only by design: the LLM runs in a separate vLLM (GPU) or Ollama (CPU)
# container and is reached over its OpenAI-compatible HTTP API, so this image
# installs only the serving dependencies from requirements.txt.
#
#   docker build -t text2sql-api .
#   docker run --rm -p 8000:8000 \
#     -e T2S_LLM_BASE_URL=http://host.docker.internal:8001/v1 \
#     -v "$PWD/data:/app/data:ro" text2sql-api
# =============================================================================
ARG PYTHON_VERSION=3.11

# -----------------------------------------------------------------------------
# Stage 1 - builder: install dependencies into an isolated virtualenv
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copy only the requirements first so the dependency layer stays cached
# until requirements.txt changes.
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# -----------------------------------------------------------------------------
# Stage 2 - runtime: slim image, non-root user, no build tooling
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG APP_VERSION=dev
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="text2sql-api" \
      org.opencontainers.image.description="Text-to-SQL LLM gateway: schema RAG, LoRA adapter routing, SQL safety validation" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/Hardik1608/text-to-sql-llm-platform" \
      org.opencontainers.image.licenses="MIT"

# PYTHONPATH=/app makes both `src` and `api` importable (api.main imports src.*).
# XDG_CACHE_HOME points library caches at /tmp because the compose service runs
# with a read-only root filesystem.
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/tmp/.cache \
    T2S_DATA_DIR=/app/data

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home \
      --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
# Application code stays root-owned (read-only for the runtime user).
COPY src/ ./src/
COPY api/ ./api/

# Mount points: ./data -> /app/data (Spider schemas + SQLite DBs, read-only),
# ./adapters -> /adapters (LoRA adapter manifests, read-only).
RUN mkdir -p /app/data /adapters

# Numeric UID:GID so Kubernetes `runAsNonRoot` can verify it without /etc/passwd.
USER 10001:10001
EXPOSE 8000

# Liveness only (GET /health never touches the LLM backend). Readiness is
# GET /ready, which returns 503 until the LLM backend is reachable.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# Single worker on purpose: prometheus_client keeps metrics per process, so
# scale out with more containers rather than more uvicorn workers.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
