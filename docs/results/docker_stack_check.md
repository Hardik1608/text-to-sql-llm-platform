# Docker Compose stack: end-to-end check (2026-10-12)

Host: Ubuntu 22.04, RTX 3060 12 GB, Docker 29.7, Compose 5.5, NVIDIA container toolkit 1.20.
Command: `docker compose --profile gpu up -d --build`, with `T2S_ADAPTERS=spider-v1-e3=/adapters/spider-v1-e3`.

| Check | Result |
|---|---|
| Containers | api, vllm, redis, redis-exporter, prometheus, grafana: all **healthy** |
| `/health`, `/ready` | `{"status":"ok"}` · `{"llm":true,"cache":true,"databases":206}` |
| `/query` in all 4 modes | HTTP 200, correct rows. `prompt`/`rag` → base model; `qlora`/`qlora_rag` → adapter `spider-v1-e3` (vLLM in Docker) |
| Generation latency through the container network | 310–402 ms for one question |
| Cache | repeated question: `cached=true`, 1 ms end to end (Redis) |
| Routing without `db_id` | routed to `pets_1`, correct result |
| Unsafe request ("Delete all singers…") | the model answered `CANNOT_ANSWER`; nothing executed |
| Prometheus | 4/4 targets up (api, vllm, redis, self); 8 alert rules loaded; `sum(text2sql_requests_total)` = 7 |
| Grafana 12.4 | datasource OK; dashboard "Text2SQL Overview" provisioned |
| Container hardening | API runs as UID 10001; root filesystem is read-only |
| GPU memory | 10.1 GB (vLLM) |

Bug found by this run: `numpy` was missing from `requirements.txt`. It used to arrive through
`rank-bm25`, which was later replaced. Local venvs still had numpy installed, so all tests passed,
but the clean Docker image crashed on import. Fixed. A fresh-venv test run now gives 131 passed
and 3 skipped (the real-Spider tests skip without data, as in CI).
