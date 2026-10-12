# Benchmark - Spider test (n=2147)

- Base model: `Qwen/Qwen2.5-3B-Instruct`; adapter: `spider-v1-e3`; prompt version `v2`; repairs: 0
- LLM backend: `http://localhost:8001/v1` (src/serving/hf_server.py, RTX 3060, bf16, sequential); run at 20261009-170217

| Approach | Exec Acc | Exact Match | Valid SQL | Hallucination | P50 latency | P95 latency | Prompt tok |
|---|---|---|---|---|---|---|---|
| Base + Prompting | 63.4% | 29.0% | 87.4% | 6.9% | 745 ms | 1897 ms | 577 |
| Base + Schema RAG | 66.9% | 38.5% | 88.0% | 7.2% | 678 ms | 1863 ms | 491 |
| QLoRA Fine-Tuned | 72.9% | 50.1% | 92.9% | 7.0% | 821 ms | 2334 ms | 354 |
| QLoRA + Schema RAG | 76.1% | 50.0% | 93.0% | 6.9% | 883 ms | 2489 ms | 491 |

Execution accuracy by Spider hardness:

| Approach | Easy | Medium | Hard | Extra |
|---|---|---|---|---|
| Base + Prompting | 79.1% (n=468) | 68.3% (n=853) | 50.6% (n=480) | 47.7% (n=346) |
| Base + Schema RAG | 84.4% (n=468) | 69.9% (n=853) | 57.5% (n=480) | 48.8% (n=346) |
| QLoRA Fine-Tuned | 87.4% (n=468) | 77.7% (n=853) | 62.7% (n=480) | 55.8% (n=346) |
| QLoRA + Schema RAG | 89.1% (n=468) | 80.1% (n=853) | 68.5% (n=480) | 59.5% (n=346) |
