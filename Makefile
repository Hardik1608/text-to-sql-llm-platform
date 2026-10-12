# =============================================================================
# Text-to-SQL LLM platform - developer commands. Run `make help`.
# Recipes use POSIX sh: on Windows run make from Git Bash or WSL.
# =============================================================================
.DEFAULT_GOAL := help

VENV ?= .venv
ifeq ($(OS),Windows_NT)
  VENV_PY := $(VENV)/Scripts/python.exe
  SYS_PY  ?= python
else
  VENV_PY := $(VENV)/bin/python
  SYS_PY  ?= python3
endif
# Use the project venv once `make setup` has created it.
PYTHON ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),$(SYS_PY))

COMPOSE      ?= docker compose
ALL_PROFILES := --profile gpu --profile cpu --profile tracking
SRC_DIRS     := src api tests

API_PORT     ?= 8000
API_URL      ?= http://localhost:$(API_PORT)
MODE         ?= prompt
OLLAMA_MODEL ?= qwen2.5:3b
SERVICE      ?=
EVAL_ARGS    ?=

.PHONY: help setup lint format test up up-gpu up-cpu up-tracking down ps logs eval smoke validate

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------------------------------------------------------------- local dev
setup: ## Create .venv, install runtime + dev deps, create .env from .env.example
	$(SYS_PY) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt -r requirements-dev.txt
	@test -f .env || { cp .env.example .env && echo "Created .env from .env.example"; }

lint: ## Ruff lint + format check (same checks as CI)
	$(PYTHON) -m ruff check $(SRC_DIRS)
	$(PYTHON) -m ruff format --check $(SRC_DIRS)

format: ## Auto-fix lint findings and format the code
	$(PYTHON) -m ruff check --fix $(SRC_DIRS)
	$(PYTHON) -m ruff format $(SRC_DIRS)

test: ## Run the test suite with coverage
	$(PYTHON) -m pytest -q --cov=src --cov=api --cov-report=term-missing

eval: ## Benchmark the four approaches on Spider dev (EVAL_ARGS="..." for extra flags)
	$(PYTHON) -m src.evaluation.compare_approaches --split dev $(EVAL_ARGS)

# ---------------------------------------------------------------- docker stack
# Bind-mounted dirs are created up front: if Docker creates them they end up
# root-owned on Linux and training can no longer write adapters/.
data adapters:
	@mkdir -p $@

up: | data adapters ## Core stack only (api, redis, monitoring); LLM backend from .env
	$(COMPOSE) up -d --build

up-gpu: | data adapters ## Core stack + vLLM on an NVIDIA GPU (all four modes)
	T2S_LLM_BASE_URL=http://vllm:8000/v1 T2S_BASE_MODEL= $(COMPOSE) --profile gpu up -d --build
	@echo "vLLM is loading the model (first run downloads ~6 GB): $(COMPOSE) logs -f vllm"

up-cpu: | data adapters ## Core stack + Ollama on CPU (prompt/rag modes; OLLAMA_MODEL=...)
	T2S_LLM_BASE_URL=http://ollama:11434/v1 T2S_BASE_MODEL=$(OLLAMA_MODEL) OLLAMA_MODEL=$(OLLAMA_MODEL) 	  $(COMPOSE) --profile cpu up -d --build
	@echo "First run pulls $(OLLAMA_MODEL): $(COMPOSE) logs -f ollama-pull"

up-tracking: ## Add the MLflow tracking server (http://localhost:5000)
	$(COMPOSE) --profile tracking up -d mlflow

down: ## Stop and remove the containers of every profile (volumes are kept)
	$(COMPOSE) $(ALL_PROFILES) down --remove-orphans

ps: ## Show service status
	$(COMPOSE) $(ALL_PROFILES) ps

logs: ## Follow logs (all services, or SERVICE=api)
	$(COMPOSE) $(ALL_PROFILES) logs -f --tail=200 $(SERVICE)

smoke: ## Smoke-test a running stack: /health, /ready and one /query (MODE=prompt)
	@echo "GET  $(API_URL)/health"
	@curl -fsS "$(API_URL)/health" && echo
	@echo "GET  $(API_URL)/ready"
	@curl -sS -o /dev/null -w "  -> HTTP %{http_code} (503 = LLM backend not reachable yet)\n" "$(API_URL)/ready"
	@echo "POST $(API_URL)/query  mode=$(MODE)"
	@curl -sS --fail-with-body -X POST "$(API_URL)/query" -H "Content-Type: application/json" 	  -d '{"question": "How many singers are there?", "db_id": "concert_singer", "mode": "$(MODE)"}' && echo

validate: ## Validate compose (all profiles), Prometheus config/rules, dashboard JSON (needs Docker)
	$(COMPOSE) config -q
	@for p in gpu cpu tracking; do $(COMPOSE) --profile $$p config -q && echo "compose profile $$p: ok"; done
	MSYS_NO_PATHCONV=1 docker run --rm --mount type=bind,source="$(CURDIR)/infra",target=/etc/prometheus,readonly 	  --entrypoint promtool "$$($(COMPOSE) config --images | grep '^prom/prometheus:')" 	  check config /etc/prometheus/prometheus.yml
	$(PYTHON) -m json.tool monitoring/grafana/dashboards/text2sql-overview.json > /dev/null && echo "dashboard JSON: ok"
