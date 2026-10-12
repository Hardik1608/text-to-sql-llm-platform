# Case Study: When Do Prompting, RAG and Fine-Tuning Pay Off? A Text-to-SQL Platform

**Author:** Hardik Soni · **Stack:** Qwen2.5-3B-Instruct, PEFT/QLoRA, TRL, vLLM, FastAPI, sqlglot, Redis, MLflow, Prometheus/Grafana

> Every result in this document comes from a run in this repo. The GPU results come from one
> RTX 3060 (12 GB) run, and the evidence is in [`results/`](results/). The benchmark harness wrote
> those files. Nothing here is estimated and presented as a result.

---

## 1. Problem

Business users ask questions in English; the answers live in relational databases. A Text-to-SQL
system must (a) understand the question, (b) map it onto an unfamiliar schema, (c) emit SQL that
is valid *and* means the right thing, and (d) never damage or leak data — LLM output is
untrusted input that will be executed against a database.

The question behind this project: **for this task, how much does each customization technique
buy (prompting, retrieval, PEFT fine-tuning), and what does each cost?** "RAG or fine-tuning?"
is usually answered with opinions; here it is answered with measurements.

## 2. Why prompting alone falls short

The prompt-mode baseline (system prompt + static few-shot examples in
`src/generation/prompts.py`) is deliberately reasonable. It still cannot fix schema linking
(`singer.Name` vs `concert.concert_Name`, join paths through bridge tables), literal grounding
(the database stores `'United States'`, the user typed "USA") or output discipline (markdown
fences, prose, a second statement).

One design decision matters here: in *every* mode the model sees the target schema. An earlier
draft gave the prompt-only mode no schema at all. That turns the comparison into a strawman,
because no model can guess unseen table names. See §16.

## 3. Why retrieval helps (and what "RAG" means for SQL)

For Text-to-SQL, "retrieval" is three separate retrievers (`src/retrieval/`), all rebuilt in
`ContextBuilder` so inference and fine-tuning share them:

| Retriever | Retrieves | Why it helps | Measured |
|---|---|---|---|
| **Schema pruning** (BM25 over table/column tokens + FK expansion) | relevant tables of the target DB | shorter, more focused prompts; mandatory for large warehouses | gold-table recall **100%** on dev / **97.6%** on train at `max_tables=6` |
| **Few-shot examples** (BM25 over *schema-masked* questions) | structurally similar (question, SQL) pairs from *other* databases | shows the SQL shape: "How many singers do we have?" retrieves "How many aircrafts do we have?" | 2.8 ms/query over 8,659 examples |
| **Value index** (n-gram lookup over DB cell values) | stored values the question mentions | grounds literals: `singer.Country = 'France'` | covers **82.5%** of gold string literals (dev) |

Two details make these work:

- **Masked similarity.** Spider's evaluation databases never appear in training. Topical
  similarity ("singers") is therefore useless, so similarity is computed after masking schema
  words and values. That leaves the structural skeleton ("how many … do we have").
- **Lucene IDF.** The textbook BM25 IDF goes to zero for terms that appear in half the
  documents. On a 4-table schema or a 2-database catalog, that silently zeroes every score
  (§16). `src/retrieval/bm25.py` uses Lucene's always-positive IDF.

**Database routing** is the same index at database level, used when the caller doesn't send a
`db_id`. Measured over all 206 Spider databases: top-1 **78.3%**, top-3 **89.0%**, at 0.5 ms.

## 4. Why and when PEFT helps

Fine-tuning changes *behaviour*, not knowledge. The adapter cannot memorise unseen databases, but
it can learn SQL conventions and how to read the schema it is shown, and it removes the need for
few-shot examples in the prompt. It cannot fix literal grounding (only retrieval sees the data),
and it adds training cost plus an artifact to version and serve. The hypothesis to test: **the
combination wins**. Fine-tuning brings behaviour and retrieval brings grounding.

| | Full schema | Retrieval-augmented |
|---|---|---|
| Base model | `prompt` | `rag` |
| QLoRA adapter | `qlora` | `qlora_rag` |

## 5. Dataset

Spider 1.0 (2024 release): 8,659 train (`train_spider` + `train_others`) / 1,034 dev / 2,147 test
questions over 206 SQLite databases. `src/ingestion/download_spider.py` renders schemas from
`tables.json` *with PRIMARY KEY and FOREIGN KEY constraints* (join paths are the most common
failure). It writes `data/processed/{split}.jsonl` and a structured `schemas.json` that every
stage shares (`src/schema.py`).

Data quality notes found while building:
- 1 of 9,693 train+dev gold queries references a table (`Ref_Company_Types`) that is absent from
  its schema.
- 2 gold queries fail to execute on their own databases.

The evaluator reports the count of gold execution errors in every run's `summary.json` instead
of silently dropping them.

**Splits and leakage.**
- **Spider test (2,147 questions, 40 databases) gives the headline results.** No decision used it.
- **200 dev questions selected the checkpoint** (§16, lesson 7), so dev results are reported as
  secondary and slightly optimistic.
- **The SFT validation split holds out 7 whole databases** (300 questions) from train. Spider is
  cross-domain, so a random split would leak schemas.

## 6. Model selection

**Qwen2.5-3B-Instruct**:
- strong code/SQL pre-training for its size;
- fits QLoRA training on a single 12 GB consumer GPU (RTX 3060) and bf16 inference with room
  left for the KV cache;
- supported by vLLM multi-LoRA serving.

A 7B model would likely score higher but doubles training time and leaves almost no KV-cache
headroom at 12 GB. For this project, the *relative* comparison between methods matters more
than the absolute score.

## 7. QLoRA configuration

| Setting | Value | Reasoning |
|---|---|---|
| Base precision | 4-bit NF4, double quantization | ~2 GB of weights; NF4 is information-optimal for normally distributed weights |
| Compute dtype | bf16 (auto) | Ampere supports it; fp16 fallback on older GPUs |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | scale = alpha / r = 2; r=16 is the usual capacity sweet spot for ~10k examples |
| Target modules | q, k, v, o, gate, up, down | all linear projections; attention-only LoRA under-fits structured output |
| Trainable params | ~30M (~1% of 3.1B) | logged by `get_nb_trainable_parameters()` |
| Loss | completion-only (SQL tokens) | prompt tokens (schema, examples) are masked; the model learns to write SQL, not to copy schemas |
| Optimizer | paged AdamW 8-bit, lr 2e-4 cosine, 3% warmup | standard QLoRA recipe; paging absorbs memory spikes |
| Effective batch | 16 (4 × 4 accumulation) | 519 steps / epoch |
| Max length | 1,536 tokens (Qwen tokenizer) | drops 61 / 8,359 over-long examples (0.7%) |
| Checkpoint selection | the final weights (see §16, lesson 7) | lowest validation loss chose a worse checkpoint |

**Measured on the 3060** (`docs/results/training_and_selection.md`):

| Measurement | Value |
|---|---|
| Peak VRAM | **6.29 GB** |
| Time per optimizer step | 19.5 s |
| Total runtime | **8.5 h** (3 epochs, 1,557 steps) |
| Validation loss | lowest at step 100 (0.204); rose to 0.401 by the end |
| Training data | 8,298 examples, mean 536 tokens |

## 8. Training pipeline

```
Spider train ──► prepare_data.py ──────────────────────────────► train_lora.py ──► adapters/spider-v1/
                 • same ContextBuilder as the API               • 4-bit base + LoRA     • adapter_model.safetensors
                 • --context mixed: 50% full-schema prompts,    • TRL SFT, completion   • adapter_manifest.json
                   50% retrieval-augmented prompts                only loss             • MLflow run (params,
                 • few-shot pool excludes the example's own DB  • MLflow tracking        loss curves, VRAM)
                 • pruned schemas always keep the gold tables
                 • validation held out by database
```

Key decisions:
- **One adapter, two serving modes.** Training on a 50/50 mix of full-schema and RAG prompts
  means `qlora` and `qlora_rag` are both in-distribution. The alternative was two adapters.
- **Leakage-aware retrieval at train time.** At evaluation, retrieved examples always come from
  *other* databases. If training prompts contained same-database examples, the model would
  learn to copy them, and that habit doesn't transfer.
- **Reproducibility.** The train file's SHA-256, prompt version and data stats are logged with
  every run. `--smoke-test` exercises the full code path (tokenization, label masking, training,
  evaluation, saving, MLflow) on CPU with a tiny Qwen2 in about a minute. **Measured:** passes
  on TRL 1.14 / transformers 5.19 / PEFT 0.21 / MLflow 3.17.

## 9. Evaluation methodology

`src/evaluation/compare_approaches.py` runs every question of a split through the **serving
pipeline** in each mode (no cache, no repair by default), then computes:

| Metric | Definition |
|---|---|
| **Execution accuracy (primary)** | predicted and gold results match: multiset comparison; order-sensitive only if gold has `ORDER BY`; tolerant to column permutations (Spider's execution-eval conventions) |
| Exact match | sqlglot-normalized structural equality (literals masked, aliases resolved, identifiers lower-cased); an approximation of Spider's component EM, reported as secondary |
| Valid SQL rate | passed the safety validator (parseable, read-only, schema-consistent) |
| Hallucination rate | unknown table/column caught by the validator |
| Unsafe-generation rate | multi-statement / write / forbidden-function output |
| Latency, tokens, cost | P50/P95 end-to-end, mean prompt tokens, $/1k queries from GPU $/h × wall time |
| By hardness | Spider's easy/medium/hard/extra, re-implemented over the AST |

**Measured checks on the metrics themselves:**
- The hardness classifier reproduces the official dev distribution within a few questions: **248/444/187/155** vs official **248/446/174/166**.
- Gold-vs-gold exact match is **100%**.

Statistics: at about 70% accuracy, the 95% CI is about ±1.9 points on test (n=2,147) and ±2.8
points on dev (n=1,034). Differences inside the interval are not claimed as wins. The modes are
scored on the same questions, so a paired test (McNemar) is the stricter check and is the next
addition to the harness.

## 10. Serving architecture

- **vLLM** serves the base model once and selects LoRA adapters *per request* through the
  OpenAI `model` field. One GPU can serve the base model and several adapter versions side by
  side (multi-tenant, A/B between adapter versions); `VLLM_MAX_LORAS` sets how many.
- **Hot-swap without restart.** `VLLM_ALLOW_RUNTIME_LORA_UPDATING` exposes
  `/v1/load_lora_adapter`. The API wraps it as `POST /adapters/load`, behind `X-Admin-Token`.
- **Adapter registry** (`src/serving/vllm_client.py`): versioned names, a default adapter, and
  the manifest (base model, rank, eval loss, data hash) exposed at `GET /adapters`.
- **Cache semantics.** Redis caches only the *generation*. Key = question + db + mode + model or
  adapter version + prompt-template version. On a hit, the SQL is still re-validated and
  re-executed, so results stay fresh and tightened safety rules apply to old entries. Redis
  failures degrade to a miss. **Measured:** a cached request took **9 ms** end to end versus
  about 7 s of uncached CPU generation.
- **Portability.** The same client runs against vLLM (production, Linux/WSL2), Ollama (CPU
  laptop) or `hf_server.py`, a transformers+PEFT reference server for native-Windows GPU
  boxes. The benchmark code doesn't change.
- **Self-repair.** If the SQL fails validation (hallucinated identifier, syntax) or execution,
  the error is fed back once. Unsafe output is never "repaired"; it is rejected and counted.

## 11. Safety engineering

LLM output is untrusted input that will be executed. Four stages, defense in depth:

| Stage | Mechanism | Blocks |
|---|---|---|
| 1 Parse | sqlglot; exactly one statement | stacked queries (`SELECT 1; DROP TABLE t`), garbage |
| 2 Read-only | root must be SELECT or a set operation; no write/DDL/session node anywhere in the AST; function denylist | DML hidden in CTEs, `SELECT INTO`, `FOR UPDATE`, `PRAGMA`, `ATTACH`, `load_extension`, sleep/file functions |
| 3 Allowlist | tables must exist in the target DB (or be CTEs); columns resolved with sqlglot's qualifier | hallucinated identifiers, cross-database references, table-valued functions |
| 4 Execution guard | read-only URI + `PRAGMA query_only`, SQLite **authorizer** (SELECT/READ/FUNCTION only), progress-handler timeout, row cap, value-length limit | anything that slipped through: the database itself refuses to write |

- **Execute what was validated.** The executor runs the *re-serialized AST*, not the raw string,
  which closes parser-differential tricks.
- **SQLite quirk handled.** `WHERE country = "France"` is a string literal in SQLite when no
  such column exists. Spider's gold SQL relies on this everywhere, and a naive column check
  rejects it.

**Measured:**
- **1 of 9,693** gold queries (0.01%) is rejected, a genuine Spider annotation error.
- Re-serialized SQL returns identical results for **100%** of executable gold queries.
- Validation takes about **4 ms**.
- Each attack class above has a unit test at both the validator and the executor layer (the
  executor tests bypass the validator).

## 12. Observability

Prometheus metrics (`src/serving/telemetry.py`, bounded label sets):
- request counter by mode × status;
- end-to-end and per-stage latency histograms (retrieval / generation / validation / execution);
- prompt and completion token counters;
- cache hit / miss / error;
- validation failures by reason (the attack and hallucination signal);
- repairs, adapter traffic, in-flight requests, build info.

All label combinations are created at startup, so dashboards show 0 rather than "No data".

On top of that:
- **Grafana dashboard** (provisioned): rates, error ratios, P50/P95/P99 by mode, stage
  breakdown, tokens/s, cache hit ratio, validation reasons, and a vLLM row (queue depth,
  KV-cache usage, TTFT, generation throughput).
- **Alert rules:** API down, P95 > 4 s, rejected+error ratio > 10%, LLM error spikes, low cache
  hit ratio, vLLM down, vLLM queue backlog. They are validated with `promtool` unit tests, and
  in a live Prometheus test all 44 dashboard queries returned data.
- **MLflow:** training runs (loss curves, peak VRAM, adapter artifact) and benchmark runs
  (`--mlflow`).

## 13. Benchmarks

**Main result — Spider test, n = 2,147, 40 unseen databases.** Measured with Qwen2.5-3B and
adapter `spider-v1-e3` on an RTX 3060, with no repairs. No decision used the test split.

| Approach | Exec Acc | Exact Match | Valid SQL | Hallucination | P95 latency | Prompt tok |
|---|---|---|---|---|---|---|
| Base + Prompting | 63.4% | 29.0% | 87.4% | 6.9% | 1.9 s | 577 |
| Base + Schema RAG | 66.9% | 38.5% | 88.0% | 7.2% | 1.9 s | 491 |
| QLoRA | 72.9% | 50.1% | 92.9% | 7.0% | 2.3 s | 354 |
| **QLoRA + Schema RAG** | **76.1%** | 50.0% | **93.0%** | 6.9% | 2.5 s | 491 |

Execution accuracy by hardness (test):

| Approach | Easy (468) | Medium (853) | Hard (480) | Extra (346) |
|---|---|---|---|---|
| Base + Prompting | 79.1% | 68.3% | 50.6% | 47.7% |
| Base + Schema RAG | 84.4% | 69.9% | 57.5% | 48.8% |
| QLoRA | 87.4% | 77.7% | 62.7% | 55.8% |
| QLoRA + Schema RAG | 89.1% | 80.1% | **68.5%** | **59.5%** |

The dev split (n=1,034) shows the same order: 60.2 / 66.1 / 73.6 / 75.2%.

**What the numbers answer:**
- **Fine-tuning gives the largest single gain:** +9.5 points over prompting. It changes behaviour, not knowledge:
  - exact match rises from 29% to 50%, so the SQL follows the expected structure;
  - valid SQL rises from 87.4% to 92.9%;
  - prompts are 39% shorter.
- **Retrieval helps both models, but less:** +3.5 points on the base model, +3.2 points on the adapter.
  - Retrieval and fine-tuning combine: the best mode uses both.
  - On test, the +3.2 points over QLoRA alone is a little outside the ±2-point interval.
  - On dev, the same gain is only +1.6 points, which is inside noise.
- **The gains grow with difficulty.** Hard questions improve by +17.9 points and extra questions
  by +11.8 points (prompting → QLoRA + RAG). Easy questions improve by +10.0 points.
- **Retrieval can hurt complex queries on the base model.** On dev, base + RAG fell from 41.3% to
  32.3% on *extra* questions. The likely cause is that retrieved examples of simpler queries pull the model toward
  the wrong structure. The adapter does not show this drop.
- The hallucination rate (unknown table or column) is flat at about 7%. Fine-tuning did not
  reduce identifier errors. The validator catches them before execution.

Interval: at n=2,147 and ~70% accuracy, the 95% confidence interval is ±1.9 points.

**Self-repair ablation** (Spider test, QLoRA + RAG, one retry). Evidence:
`results/vllm_serving_and_repair.md`.

| | No repair | One repair |
|---|---|---|
| Exec Acc | 76.1% | **78.0%** |
| Valid SQL | 93.0% | 95.2% |
| Hallucination | 6.9% | 4.6% |
| Prompt tokens (mean) | 491 | 543 |

- Feeding the validator's error back to the model fixes about a third of the hallucinated identifiers.
- Repair only touches queries that were already rejected, so it cannot break a correct answer.
- The cost is 11% more tokens on average.
- Hard and extra questions gain as much as easy ones (+1.7 to +2.0 points per bucket).

**Serving on vLLM** (500 dev questions, one RTX 3060):

| Parallel requests | Throughput | P50 | P95 | Exec Acc |
|---|---|---|---|---|
| 1 | 1.22 q/s | 610 ms | 1.81 s | 75.0% |
| 4 | 3.19 q/s | 952 ms | 2.77 s | 75.2% |
| 16 | 6.08 q/s | 1.91 s | 5.94 s | 74.6% |

- Continuous batching gives 5× throughput at 16 parallel requests, at 3× the median latency.
- Accuracy is the same across concurrency levels, and the same as on the reference server.

## 14. Failure modes

The failure classes the system is designed against, with the mitigation for each. The measured
rates are in §3 (value-hint coverage) and §13 (hallucination, valid SQL, repair). A per-class
breakdown of the test errors is not done yet.

| Failure | Example | Mitigation in this system |
|---|---|---|
| Schema linking | `concert_Name` vs `Name` | PK/FK-rich schema, pruning, QLoRA |
| Join path | missing bridge table | FK expansion in pruning; FK lines in prompts |
| Literal mismatch | `'USA'` vs `'United States'` | value hints (82.5% literal coverage) |
| Aggregation semantics | "the most" without `LIMIT 1`, ties | retrieved structural examples |
| Nested / set operations | `INTERSECT` vs `AND` | examples; fine-tuning (watch the *extra* bucket) |
| Hallucinated identifier | `nam` | validator → one repair attempt |
| Unsafe output / injection | `…; DROP TABLE` | rejected at stage 1–2, counted, never repaired |
| Infrastructure | LLM down, slow query | 503 with `llm_error`, retries with backoff; query timeout; cache degrades to miss |

## 15. Cost / latency trade-offs

- **Serving capacity, measured.** One RTX 3060 on vLLM serves **6.1 questions/s**
  (about 22,000/hour) at 16 parallel requests, with P95 5.9 s end to end.
  - For an interactive SLO of P95 < 2 s, keep concurrency at 1–2 per GPU: about 1.2–2 q/s.
  - The trade-off is a choice between latency and cost, and the operator picks it per tenant.
- **Self-repair:** +1.9 points of accuracy for +11% tokens and a second LLM call on about 5–7% of
  questions. That's a good price when wrong answers cost more than tokens.

- **Retrieval overhead** is negligible next to generation: routing 0.5 ms, examples 2.8 ms,
  value hints 0.3 ms, validation about 4 ms.
- **Tokens, measured on test:** prompting 577, RAG 491, QLoRA **354**, QLoRA + RAG 491 mean
  prompt tokens. RAG adds about 3 retrieved examples but prunes the schema. The adapter learned
  the conventions that few-shot examples otherwise teach, so it needs the shortest prompt.
- **Training cost.** One overnight run on a consumer GPU (8.5 h, 6.29 GB peak VRAM), amortized over every
  query after it. Retraining is needed only when SQL conventions change, not when data
  changes; that's retrieval's job.
- **Serving cost.** LoRA adapters add almost nothing in vLLM: same base weights, a small
  per-request LoRA compute. Multi-adapter serving means one GPU, not one per variant.
- **Cache.** A hit removes generation entirely (9 ms vs seconds).
- **Benchmark output.** `compare_approaches --gpu-cost-per-hour` turns wall time into $/1k
  queries for each mode.

## 16. Lessons learned (real ones, from building this)

1. **The obvious benchmark design was a strawman.** The first draft gave prompt mode no schema
   and served QLoRA without one, on a cross-database benchmark. Both would score near zero. The
   2×2 design (everything sees the schema; *retrieval* and *adapter* are the only variables)
   is what makes the comparison meaningful.
2. **Test-set leakage hides in prompts.** The original static few-shot example ("How many
   singers are there?" on `concert_singer`) was essentially dev question #1. All static
   examples now use a synthetic schema.
3. **BM25 breaks on small corpora.** With 2 databases, Robertson IDF is exactly 0, and routing
   silently returned the first database. A unit test with a 2-document corpus caught it;
   Spider's 206-database catalog had hidden it.
4. **Validate the validator on real data.** The SQLite double-quoted literal quirk would have
   rejected a large share of correct, Spider-style SQL. Running all 9,693 gold queries through
   the validator found it and gave the 0.01% false-rejection figure.
5. **Choose hyperparameters on train, not dev.** Pruning recall was 100% on dev but 97.6% on
   train, whose schemas are larger. Tuning on dev would have overstated retrieval quality.
6. **Library drift is a production risk.** TRL 1.x and transformers 5 removed `warmup_ratio`.
   It was ignored silently, which means no warmup. MLflow 3 rejects the `./mlruns` store. A
   2-step CPU smoke test caught both before any GPU time was spent.
7. **Validation loss chose the wrong checkpoint.** Loss on the held-out databases was lowest at
   step 100 (0.204) and doubled by epoch 3 (0.401).
   - On the same 200 dev questions, execution accuracy rose from 67.5% to 75.5% between those checkpoints.
   - Valid SQL rose from 86.5% to 92%.
   - Cross-entropy punishes confident tokens, so a model that writes correct SQL with more
     certainty can get a *higher* loss.
   - The fix: select checkpoints on a task metric. `train_lora.py` now keeps the final weights.
     The next step is execution accuracy on the validation databases at each checkpoint.
   - The choice used 200 dev questions, so the headline numbers come from the untouched test split.
8. **Token counting broke silently on transformers 5.** `apply_chat_template(tokenize=True)` now
   returns a dict, so every example "had" 2 tokens and the length filter dropped nothing. The
   bug showed up on the first GPU run, in the stats file, before any training time was spent.
   Fix: `return_dict=True` + `["input_ids"]`.

## 17. What changes at 10× traffic

| Pressure | Change |
|---|---|
| GPU throughput | horizontal vLLM replicas behind a router that keeps **adapter affinity** (requests for `spider-v2` go to replicas that hold it); autoscale on `vllm:num_requests_waiting` rather than CPU |
| Prefill cost | turn on vLLM **automatic prefix caching**, and reorder RAG prompts to `[system][schema][examples][question]` so prompts on the same database share a cached prefix. Measure the accuracy cost of moving examples after the schema |
| Hot adapters | `--max-loras` sized to the active tenant set; preload by traffic; LRU-evict cold adapters |
| Cache | Redis Cluster; semantic (embedding) cache for paraphrases, guarded by re-validation |
| Databases | read replicas; per-tenant connection pools; async execution queue for long analytical queries; result-size quotas |
| Safety and abuse | per-tenant rate limits and token budgets; alerting on spikes in validation-failure reasons (injection campaigns) |
| Retrieval | schemas with hundreds of tables: replace lexical pruning with a learned schema linker or embedding retrieval (recall already drops to 97.6% on Spider's larger schemas) |
| Ops | SLOs per mode (e.g. P95 < 2 s for `qlora`); canary new adapters with a hash-based traffic split before promotion; shadow-evaluate on logged traffic |
