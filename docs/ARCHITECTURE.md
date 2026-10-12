# Architecture

## 1. Request path (`POST /query`)

```mermaid
sequenceDiagram
    autonumber
    participant U as Client
    participant A as FastAPI gateway
    participant C as Redis (generation cache)
    participant R as ContextBuilder
    participant L as vLLM (base + LoRA)
    participant V as SQL validator
    participant E as Guarded executor
    U->>A: question, db_id?, mode, adapter?
    alt no db_id
        A->>R: BM25 database routing
    end
    A->>C: GET key(question, db, mode, model/adapter, prompt version)
    alt cache hit
        C-->>A: cached SQL (generation only)
    else miss
        A->>R: prune schema (BM25 + FK expansion), retrieve examples, value hints
        R-->>A: context (mode-dependent)
        A->>L: chat completion, model = base id or adapter name
        L-->>A: raw text (+ token usage)
        A->>A: extract SQL
    end
    A->>V: parse → read-only → schema allowlist
    alt hallucinated identifier / syntax error (≤ 1 repair)
        A->>L: previous answer + validator feedback
    else unsafe (multi-statement, write, forbidden function)
        A-->>U: status=rejected, violations
    end
    A->>E: re-serialized AST on a read-only connection (authorizer, timeout, row cap)
    E-->>A: rows or database error (→ ≤ 1 repair)
    A->>C: SET generation (status ok)
    A-->>U: SQL, rows, per-stage timings, token usage
```

Every step increments Prometheus metrics, for example:

- `text2sql_stage_latency_seconds{stage=...}`
- `text2sql_validation_failures_total{reason=...}`
- `text2sql_requests_total{mode, status}`

## 2. Training and evaluation pipeline

```mermaid
flowchart LR
    S[(Spider zip<br/>206 SQLite DBs)] --> I[download_spider.py<br/>PK/FK-rich schemas]
    I --> P[(data/processed<br/>train/dev/test.jsonl<br/>schemas.json)]
    P --> D[prepare_data.py<br/>same ContextBuilder as API<br/>mixed full/RAG prompts<br/>val held out by database]
    D --> F[(data/finetune<br/>train/val.jsonl)]
    F --> T[train_lora.py<br/>4-bit NF4 + LoRA r16<br/>completion-only loss]
    T --> AD[(adapters/spider-v1<br/>+ adapter_manifest.json)]
    T -.-> M[(MLflow<br/>params, curves, VRAM)]
    AD --> VL[vLLM / hf_server]
    P --> CMP[compare_approaches.py<br/>4 modes through the<br/>production pipeline]
    VL --> CMP
    CMP --> REP[(reports/run<br/>summary.md/json<br/>predictions_*.jsonl)]
    CMP -.-> M
    REP --> NB[benchmark_analysis.ipynb]
```

## 3. Deployment topology (`docker compose`)

```mermaid
flowchart TB
    subgraph core [core services]
        API[api :8000<br/>FastAPI, non-root,<br/>read-only FS]
        RED[redis :6379<br/>LRU 256 MB]
        REX[redis-exporter]
        PROM[prometheus :9090<br/>+ alert rules]
        GRAF[grafana :3000<br/>provisioned dashboard]
    end
    subgraph gpu [profile gpu]
        VLLM[vllm :8001<br/>Qwen2.5-3B + LoRA<br/>runtime hot-swap]
    end
    subgraph cpu [profile cpu]
        OLL[ollama :11434<br/>dev backend]
    end
    subgraph tracking [profile tracking]
        MLF[mlflow :5000<br/>sqlite backend]
    end
    API --> RED
    API -- OpenAI API --> VLLM
    API -. alternative .-> OLL
    PROM -- scrape --> API
    PROM -- scrape --> VLLM
    PROM -- scrape --> REX --> RED
    GRAF --> PROM
    VOLD[(./data ro)] --- API
    VOLA[(./adapters ro)] --- API
    VOLA --- VLLM
```

## 4. Module map

| Module | Responsibility |
|---|---|
| `src/schema.py` | `DbSchema` / `SchemaCatalog`: one schema model for retrieval, prompts, validation and execution |
| `src/retrieval/bm25.py` | BM25 with Lucene IDF + inverted index |
| `src/retrieval/schema_retriever.py` | database routing, table ranking and pruning |
| `src/retrieval/example_retriever.py` | schema-masked few-shot retrieval |
| `src/retrieval/value_index.py` | question phrases → stored cell values |
| `src/retrieval/context.py` | what each mode puts in the prompt (shared by API, eval, SFT data) |
| `src/generation/prompts.py` | versioned templates for the 4 modes; SFT example format |
| `src/generation/postprocess.py` | robust SQL extraction (never hides a second statement) |
| `src/safety/sql_validator.py` | stages 1–3 |
| `src/safety/executor.py` | stage 4 |
| `src/serving/vllm_client.py` | OpenAI-compatible client with retries; LoRA adapter registry; hot-swap |
| `src/serving/cache.py` | Redis / in-memory generation cache |
| `src/serving/telemetry.py` | Prometheus metric contract |
| `src/serving/hf_server.py` | reference server for Windows GPU boxes |
| `src/pipeline.py` | orchestration: cache → retrieve → generate → validate → repair → execute |
| `src/evaluation/` | metrics (EX, EM, hardness) and the 4-way benchmark |
| `src/finetuning/` | SFT data preparation and QLoRA training |
| `api/main.py` | HTTP layer, admin endpoints, readiness |
