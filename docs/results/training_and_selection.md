# QLoRA training run and checkpoint selection (RTX 3060 12 GB, Ubuntu 22.04)

From `adapters/spider-v1/adapter_manifest.json` and `runs/spider-v1/checkpoint-1557/trainer_state.json`.

| Setting / measurement | Value |
|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct, 4-bit NF4 + double quantization, bf16 compute |
| LoRA | r=16, alpha=32, dropout 0.05, q/k/v/o/gate/up/down; 29,933,568 trainable params (0.96%) |
| Data | 8,298 train / 300 val (7 held-out databases); mean 536 tokens, p95 952; train data sha256 6c672ef61ebe775c |
| Schedule | 3 epochs, 1,557 steps, effective batch 16, lr 2e-4 cosine |
| Runtime | 30,530 s (8.5 h) |
| Peak VRAM | 6.29 GB |
| Stack | torch 2.14.1+cu130, transformers 5.19.0, trl 1.14.2, peft 0.21.2, bitsandbytes 0.50.2 |

Validation loss by step:

| Step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 1100 | 1200 | 1300 | 1400 | 1500 | 1557 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Val loss | 0.2035 | 0.2307 | 0.2221 | 0.2401 | 0.2332 | 0.3096 | 0.2951 | 0.3196 | 0.3111 | 0.3387 | 0.3715 | 0.3936 | 0.3982 | 0.4009 | 0.4007 | 0.4005 |

Selection check, the same 200 dev questions (seed 7), execution accuracy:

| Adapter | QLoRA | QLoRA + Schema RAG | Valid SQL | Hallucination |
|---|---|---|---|---|
| spider-v1 (step 100, lowest val loss) | 67.5% | 71.0% | 86.5% / 87.0% | 13.5% / 13.0% |
| spider-v1-e3 (step 1557, final) | 75.5% | 76.0% | 92.0% / 92.0% | 8.0% / 8.0% |

Conclusion: validation loss rose 2x while execution accuracy rose 5-8 points. The final weights (`spider-v1-e3`) are the served adapter. `train_lora.py` now keeps the last weights (`load_best_model_at_end=False`).
