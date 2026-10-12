# Benchmark - Spider dev (n=1034)

- Base model: `Qwen/Qwen2.5-3B-Instruct`; adapter: `spider-v1-e3`; prompt version `v2`; repairs: 0
- LLM backend: `http://localhost:8001/v1` (src/serving/hf_server.py, RTX 3060, bf16, sequential); run at 20261009-155724
- Note: 200 dev questions (seed 7) were used to choose spider-v1-e3 over spider-v1, so dev is slightly optimistic. Test is the headline.

| Approach | Exec Acc | Exact Match | Valid SQL | Hallucination | P50 latency | P95 latency | Prompt tok |
|---|---|---|---|---|---|---|---|
| Base + Prompting | 60.2% | 29.7% | 80.5% | 10.2% | 706 ms | 1781 ms | 585 |
| Base + Schema RAG | 66.1% | 37.5% | 85.9% | 7.7% | 661 ms | 1711 ms | 503 |
| QLoRA Fine-Tuned | 73.6% | 54.0% | 92.8% | 7.2% | 746 ms | 2159 ms | 361 |
| QLoRA + Schema RAG | 75.2% | 54.9% | 91.8% | 8.2% | 822 ms | 2202 ms | 503 |

Execution accuracy by Spider hardness:

| Approach | Easy | Medium | Hard | Extra |
|---|---|---|---|---|
| Base + Prompting | 75.8% (n=248) | 61.3% (n=444) | 52.4% (n=187) | 41.3% (n=155) |
| Base + Schema RAG | 85.5% (n=248) | 71.0% (n=444) | 57.2% (n=187) | 32.3% (n=155) |
| QLoRA Fine-Tuned | 91.1% (n=248) | 75.9% (n=444) | 70.0% (n=187) | 43.2% (n=155) |
| QLoRA + Schema RAG | 93.2% (n=248) | 79.7% (n=444) | 67.4% (n=187) | 43.2% (n=155) |
