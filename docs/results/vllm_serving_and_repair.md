# vLLM serving benchmark and self-repair ablation (RTX 3060 12 GB, 2026-10-12)

Setup: vLLM 0.31.0 serves `Qwen/Qwen2.5-3B-Instruct` (bf16) + LoRA `spider-v1-e3`, with
`--max-lora-rank 16 --max-loras 1 --max-model-len 4096 --gpu-memory-utilization 0.85 --generation-config vllm`
and `VLLM_USE_FLASHINFER_SAMPLER=0`. GPU memory in use: 10.1 GB (5.85 GB weights, 2.25 GB KV cache,
0.81 GB CUDA graphs). Mode: QLoRA + Schema RAG.

Latency is end to end through the full pipeline: retrieval, generation, validation and SQLite
execution. The client runs on the same machine.

## 1. Throughput vs concurrency (500 dev questions, seed 7, no repair)

| Parallel requests | Throughput | P50 latency | P95 latency | Exec Acc |
|---|---|---|---|---|
| 1 | 1.22 q/s | 610 ms | 1.81 s | 75.0% |
| 4 | 3.19 q/s | 952 ms | 2.77 s | 75.2% |
| 16 | **6.08 q/s** | 1.91 s | 5.94 s | 74.6% |

- Continuous batching gives 5.0x throughput at 16 parallel requests; P50 latency rises 3.1x.
- Accuracy stays the same at all concurrency levels (74.6–75.2%). It also matches the
  reference-server result on the full dev set (75.2%), so the vLLM setup is correct.
- One stream on vLLM is faster than on the sequential reference server: P50 610 ms vs 822 ms.

Reports: `reports/vllm_c1/20261012-130834_dev`, `reports/vllm_c4/20261012-131526_dev`,
`reports/vllm_c16/20261012-131804_dev` (on the GPU box).

## 2. Self-repair ablation (Spider test, n=2,147, QLoRA + Schema RAG)

| | Repairs 0 (reference server, 2026-10-09) | **Repairs 1 (vLLM, 2026-10-12)** | Change |
|---|---|---|---|
| Exec Acc | 76.1% | **78.0%** | +1.9 |
| Exact Match | 50.0% | 51.6% | +1.6 |
| Valid SQL | 93.0% | 95.2% | +2.2 |
| Hallucination | 6.9% | 4.6% | -2.3 |
| Easy / Medium / Hard / Extra | 89.1 / 80.1 / 68.5 / 59.5% | 90.8 / 82.1 / 70.2 / 61.3% | +1.7 / +2.0 / +1.7 / +1.8 |
| Mean prompt tokens | 491 | 543 | +11% |

- One retry fixes about a third of the invalid queries: hallucinated identifiers drop from 6.9% to 4.6%.
- A repair runs only after the validator or the database rejects a query, and those queries were
  already wrong. Repair therefore cannot break a correct answer.
- The cost: 11% more prompt tokens on average, and a second LLM call for about 5–7% of questions.
- Caveat: the two runs used different serving backends (HF reference server vs vLLM). On dev,
  the two backends give the same accuracy (75.2% vs 74.6–75.2%), so the backend effect is small
  next to +1.9 points.

Report: `reports/repair/20261012-132000_test` (on the GPU box).
