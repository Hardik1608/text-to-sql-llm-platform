# Runbook: QLoRA Training & Benchmarking on an RTX 3060 (12 GB)

This is the GPU half of the project: train the `spider-v1-e3` adapter, then run the 4-way benchmark.
Everything else (API, retrieval, safety layer, evaluation harness) runs and is tested on a CPU laptop.

## 1. Will it fit? (memory plan)

| Component | Estimate | Why |
|---|---|---|
| Base weights, 4-bit NF4 + double quant | ~2.0 GB | 3.1B params × ~0.5 byte + quantization constants |
| LoRA r=16 on all 7 projections | ~30M params → ~0.5 GB with grads + 8-bit paged AdamW | 36 layers × (q,k,v,o,gate,up,down) |
| Activations (gradient checkpointing) | ~1–2 GB at batch 4 × ~700 tokens | only layer boundaries are stored |
| Logits | up to ~2.5 GB | Qwen's 151,936-token vocabulary is the largest activation |
| CUDA context / allocator slack | ~1 GB | |
| **Total** | **~7–10 GB** | fits 12 GB; RTX 30xx (Ampere) supports bf16, picked automatically |

These are planning estimates. The script measures `peak_vram_gb` and writes it to MLflow and to
`adapters/<name>/adapter_manifest.json`. The measured peak was **6.29 GB**.
If you hit an out-of-memory error: `--batch-size 2 --grad-accum 8` (same effective batch of 16).

## 2. Time budget

- Training data: **8,230** prompt/completion pairs from 139 Spider databases, about 5.6M tokens per epoch. This was estimated from character counts; `prepare_data` prints exact counts once `transformers` is installed.
- Steps: ceil(8,230 / 16) ≈ **515 optimizer steps per epoch**.
- Expect **several hours per epoch** on a 3060. Measure before committing (step 4 below) and choose the epoch count from the measured `s/it`. The published adapter used 3.

## 2b. Linux box: one script

`scripts/gpu_train.sh setup | data | smoke | probe | train | eval` runs each stage. Measured on an
RTX 3060 (Ubuntu 22.04, torch 2.14 + CUDA 13, TRL 1.14):
- **19.5 s per optimizer step** (16 sequences)
- **5.84 GB peak VRAM** (probe; ~6.4 GB during the full run)
- 1,557 steps for 3 epochs, about **8–9 h**

## 3. Option A: Windows native (simplest)

```powershell
# NVIDIA driver installed; Python 3.11
python -m venv .venv-train; .venv-train\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train.txt      # bitsandbytes has official Windows wheels (>= 0.43)

# Data (or copy data/processed + data/finetune from the laptop)
python -m src.ingestion.download_spider
python -m src.finetuning.prepare_data      # exact token counts with the Qwen tokenizer

# 1) Stack check: tiny model, CPU, 2 steps (~1 min)
python -m src.finetuning.train_lora --smoke-test --no-mlflow

# 2) Throughput probe: 30 real steps -> read s/it from the progress bar
python -m src.finetuning.train_lora --max-steps 30 --output adapters/probe --no-mlflow

# 3) The real run (overnight). Tracks to sqlite:///mlflow.db
python -m src.finetuning.train_lora --epochs 3 --output adapters/spider-v1-e3
mlflow ui --backend-store-uri sqlite:///mlflow.db     # loss curves, peak VRAM, adapter artifact
```

Benchmark on the same box with the reference server (`src/serving/hf_server.py`). vLLM is not
available on native Windows; the reference server exposes the same OpenAI-compatible API,
including per-request LoRA selection:

```powershell
# Terminal 1 - base model + adapter, ~6.5 GB bf16 (add --load-in-4bit to save VRAM)
python -m src.serving.hf_server --adapter spider-v1-e3=adapters/spider-v1-e3 --port 8001

# Terminal 2 - 4-way benchmark (sequential: the reference server does not batch)
$env:T2S_LLM_BASE_URL = "http://localhost:8001/v1"
python -m src.evaluation.compare_approaches --split test --adapter spider-v1-e3 --concurrency 1 --gpu-cost-per-hour 0.20
# quick pass first:  --limit 200 --sample
```

## 4. Option B: Linux or WSL2 + vLLM (serving path)

```bash
# Ubuntu on WSL2 with the NVIDIA "CUDA on WSL" driver
pip install vllm
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-3B-Instruct \
  --enable-lora --lora-modules spider-v1-e3=adapters/spider-v1-e3 \
  --max-lora-rank 16 --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8001

T2S_LLM_BASE_URL=http://localhost:8001/v1 \
  python -m src.evaluation.compare_approaches --split test --adapter spider-v1-e3 --concurrency 8
```

You can also run `docker compose --profile gpu up -d`; see `infra/README.md`. That needs Docker
Desktop with the WSL2 backend and GPU support. vLLM batches concurrent requests, so the full
4 × 2,147-query test benchmark is much faster than on the reference server.

## 5. After the run: what to record

| Where | What | Goes into |
|---|---|---|
| `adapters/spider-v1-e3/adapter_manifest.json` | `peak_vram_gb`, `train_runtime_s`, `eval_loss`, trainable params | README "Training" table |
| `reports/<run>/summary.md` | EX / EM / validity / latency for all 4 modes, by hardness | README "Benchmark Results", case study §13 |
| MLflow `text-to-sql-qlora` | loss curves (screenshot) | case study §8 |

Report headline numbers from the test split only. Dev questions were used to choose the
checkpoint, so dev results are slightly optimistic.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | `--batch-size 2 --grad-accum 8`; or `--max-seq-len 1024` (drops ~2% of examples) |
| `bitsandbytes` import error on Windows | `pip install -U bitsandbytes` (>= 0.43 ships Windows wheels); check `nvidia-smi` works |
| MLflow `filesystem tracking backend ... maintenance mode` | already handled: default URI is `sqlite:///mlflow.db` |
| Validation loss rises while training continues | not necessarily over-fitting: on this task the lowest-loss checkpoint scored 8 points *lower* on execution accuracy ([`training_and_selection.md`](results/training_and_selection.md)). Compare checkpoints on execution accuracy, not loss |
| vLLM: `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | FlashInfer compiles its sampler on first use and needs the CUDA toolkit. Set `VLLM_USE_FLASHINFER_SAMPLER=0` to use the PyTorch sampler, which gives identical results with greedy decoding. |
| Adapter works in the HF server but vLLM rejects it | `--max-lora-rank` must be >= the training rank (16) |
