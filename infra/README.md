# Infrastructure: deployment, observability, CI

Everything runs from `docker-compose.yml` at the repo root. The core stack is
always on, and the LLM backend is picked with a compose profile. The `make`
targets wrap the commands (run `make help`; on Windows use Git Bash or WSL).

```bash
make setup          # .venv + deps, and .env copied from .env.example
make up-gpu         # or: make up-cpu
make smoke          # GET /health, GET /ready, one POST /query
make down           # stop every profile (named volumes are kept)
```

## Profiles

| Command | Adds | Use it for |
|---|---|---|
| `make up` | core only: `api`, `redis`, `redis-exporter`, `prometheus`, `grafana` | API against an LLM server you run elsewhere (set `T2S_LLM_BASE_URL`, e.g. `http://host.docker.internal:8001/v1`) |
| `make up-gpu` | `vllm` (Qwen2.5-3B-Instruct, multi-LoRA) | The full platform: all four modes `prompt`, `rag`, `qlora`, `qlora_rag` |
| `make up-cpu` | `ollama` + one-shot `ollama-pull` (`qwen2.5:3b`) | CPU-only dev. **`prompt` and `rag` only**: Ollama cannot serve LoRA adapters, so `qlora*` modes need vLLM |
| `make up-tracking` | `mlflow` | Experiment and adapter tracking for training/eval runs on the host |

The `make up-gpu` and `make up-cpu` targets point the API at the matching
backend for you. With plain compose, set it in `.env` yourself:

```bash
docker compose --profile gpu up -d --build
# CPU: first set T2S_LLM_BASE_URL=http://ollama:11434/v1 and T2S_BASE_MODEL=qwen2.5:3b in .env
docker compose --profile cpu up -d --build
```

**GPU prerequisites:** an NVIDIA GPU with Docker GPU support (Docker Desktop with
the WSL2 backend on Windows, or the NVIDIA Container Toolkit on Linux).
`vllm/vllm-openai:v0.31.0` is a CUDA 13 build and needs driver R580 or newer. With
an older driver, set `VLLM_IMAGE=vllm/vllm-openai:v0.31.0-cu129`. The first boot
downloads about 6 GB of weights (cached in the `hf-cache` volume). Follow it with
`docker compose logs -f vllm`. `GET /ready` returns 200 once vLLM answers.

## Ports

| Service | URL | Notes |
|---|---|---|
| API | http://localhost:8000 | `/docs`, `/health` (liveness), `/ready` (LLM reachable), `/metrics` |
| vLLM | http://localhost:8001/v1 | OpenAI-compatible; engine metrics on `/metrics` |
| Ollama | http://localhost:11434/v1 | profile `cpu` |
| Grafana | http://localhost:3000 | anonymous read-only; admin login from `.env` |
| Prometheus | http://localhost:9090 | `/targets`, `/alerts` |
| MLflow | http://localhost:5000 | profile `tracking` |
| Redis | 127.0.0.1:6379 | loopback only (no auth) |

Change any host port with its `*_PORT` variable in `.env`.

## LoRA adapters at runtime

Each adapter is a PEFT output directory at `./adapters/<name>/` holding
`adapter_config.json` and `adapter_model.safetensors`. It is mounted read-only at
`/adapters/<name>` in both the `api` and `vllm` containers, so one path works in
both.

**At startup:** every `name=path` pair in `T2S_ADAPTERS` (default
`spider-v1=/adapters/spider-v1`) is passed to vLLM as `--lora-modules`. An
adapter that is not on disk yet is skipped with a log line instead of crashing
vLLM.

**Hot-swap without a restart:** set `T2S_ADMIN_TOKEN` in `.env` first. The admin
endpoints stay disabled while it is empty.

```bash
TOKEN=...   # value of T2S_ADMIN_TOKEN

# Load ./adapters/spider-v2 into the running vLLM and register it in the API
curl -X POST localhost:8000/adapters/load -H "X-Admin-Token: $TOKEN" \
  -H "Content-Type: application/json" -d '{"name": "spider-v2", "path": "/adapters/spider-v2"}'

curl localhost:8000/adapters                       # registered adapters + default

# Route a request to that adapter version (A/B against spider-v1)
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "How many singers are there?", "db_id": "concert_singer", "mode": "qlora_rag", "adapter": "spider-v2"}'

curl -X POST localhost:8000/adapters/unload -H "X-Admin-Token: $TOKEN" \
  -H "Content-Type: application/json" -d '{"name": "spider-v2"}'
```

Under the hood the API calls vLLM `POST /v1/load_lora_adapter`
(`{"lora_name", "lora_path"}`). The compose file enables that endpoint with
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, which vLLM flags as a local-development
feature. Keep these limits in mind:

- Runtime loads live in memory only. Add the adapter to `T2S_ADAPTERS` to keep it
  across restarts.
- The adapter's rank must not exceed `VLLM_MAX_LORA_RANK` (16).
- At most `VLLM_MAX_LORAS` (4) adapters run in a single batch.

## Dashboards and alerts

- **Grafana:** http://localhost:3000 opens on the provisioned **Text2SQL
  Overview** dashboard (folder *Text2SQL*, uid `text2sql-overview`). Its rows are
  overview tiles, traffic and outcomes, latency P50/P95/P99 and per-stage
  timing, LLM token usage and errors, SQL-safety rejections and repairs,
  cache/Redis, and vLLM (queue, KV-cache, TTFT, token throughput). The `Mode`
  variable filters per approach. To change it: edit in the UI, export the JSON,
  then overwrite `monitoring/grafana/dashboards/text2sql-overview.json`. Grafana
  re-reads that folder every 30 s.
- **Prometheus:** http://localhost:9090/targets shows the scrape jobs
  (`text2sql-api`, `vllm`, `redis`, `prometheus`). The `vllm` job uses DNS
  discovery, so it has no target while the GPU profile is off, and `VLLMDown`
  only fires when vLLM is deployed but not serving.
- **Alerts:** rules are in `infra/alerts.yml`; firing ones show at
  http://localhost:9090/alerts. The stack has no Alertmanager. To route alerts,
  add one and uncomment `alerting:` in `infra/prometheus.yml`.

| Alert | Fires when | Severity |
|---|---|---|
| `Text2SQLApiDown` | API scrape fails for 1m | critical |
| `Text2SQLHighLatencyP95` | P95 end-to-end latency > 4s for 10m (per mode) | warning |
| `Text2SQLHighFailureRatio` | rejected + exec_error + llm_error > 10% of requests for 10m (min 10 req) | warning |
| `Text2SQLLLMErrorsSpike` | > 5 LLM errors of one kind within 5m | critical |
| `Text2SQLLowCacheHitRatio` | cache hit ratio < 10% for 30m (min 50 lookups) | warning |
| `RedisDown` | Redis unreachable for 2m | warning |
| `VLLMDown` | vLLM target down for 3m | critical |
| `VLLMQueueBacklog` | > 10 requests waiting in vLLM for 5m | warning |

After editing `prometheus.yml` or `alerts.yml`, validate and reload:
`make validate`, then `docker compose kill -s SIGHUP prometheus`.

## MLflow

Run `make up-tracking`, then `export MLFLOW_TRACKING_URI=http://localhost:5000`
before running training or evaluation on the host. Runs are stored in SQLite and
artifacts are proxied by the server, both in the `mlflow-data` volume.

## CI (`.github/workflows/ci.yml`)

Runs on pushes to `main` and on pull requests. Superseded runs are cancelled and
the jobs only read the repo.

- **`lint`:** `ruff check` and `ruff format --check` on `src api tests`.
- **`test`:** pytest with coverage on Python 3.11 and 3.12. `coverage.xml` is
  uploaded as an artifact.
- **`config`:** `docker compose config` for every profile, promtool checks on the
  Prometheus config and rules (using the same image as the stack), and a
  structure check of the Grafana JSON.
- **`docker`:** builds the API image with the GHA layer cache (no push), then
  checks that it runs as non-root and that `api.main` imports.

Run the same checks locally with `make lint test validate`.

## Troubleshooting

- **`/ready` returns 503:** the backend is still loading, or `T2S_LLM_BASE_URL`
  or `T2S_BASE_MODEL` doesn't match what the backend serves.
- **`skip LoRA ... adapter_config.json not found` in the vLLM log:** the
  `T2S_ADAPTERS` path doesn't point at the directory that holds
  `adapter_config.json`.
- **CUDA or driver errors when vLLM starts:** use the `-cu129` image (see GPU
  prerequisites).
- **The API container is hardened:** it runs as non-root with a read-only root
  filesystem; only `/tmp` is writable.
