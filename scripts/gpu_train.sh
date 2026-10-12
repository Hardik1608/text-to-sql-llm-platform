#!/usr/bin/env bash
# GPU workflow for the QLoRA run on a Linux box (tested: Ubuntu 22.04, RTX 3060 12 GB).
#
#   scripts/gpu_train.sh setup   # Python 3.11 venv (via uv) + CUDA torch + training deps
#   scripts/gpu_train.sh data    # Spider ingestion + SFT data with exact Qwen token counts
#   scripts/gpu_train.sh smoke   # tiny model, 2 steps: checks the software stack
#   scripts/gpu_train.sh probe   # 30 real QLoRA steps: s/step + peak VRAM -> time estimate
#   scripts/gpu_train.sh train   # full run in tmux session "t2s-train" (EPOCHS=3 by default)
#   scripts/gpu_train.sh eval    # serve base + adapter and run the 4-way benchmark
#
# Env: EPOCHS (3), ADAPTER (spider-v1), VENV (.venv-train), EVAL_ARGS (extra benchmark flags)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
VENV=${VENV:-.venv-train}
PY="$VENV/bin/python"
EPOCHS=${EPOCHS:-3}
ADAPTER=${ADAPTER:-spider-v1}
mkdir -p logs

case "${1:-}" in
  setup)
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$PY" torch   # Linux wheels bundle CUDA
    uv pip install --python "$PY" -r requirements-train.txt -r requirements-dev.txt
    "$PY" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    ;;
  data)
    "$PY" -m src.ingestion.download_spider
    "$PY" -m src.finetuning.prepare_data 2>&1 | tee "logs/prepare_data.log"
    ;;
  smoke)
    "$PY" -m src.finetuning.train_lora --smoke-test --no-mlflow --output runs/smoke-adapter 2>&1 | tee logs/smoke.log
    ;;
  probe)
    "$PY" -m src.finetuning.train_lora --max-steps 30 --eval-steps 30 --save-steps 30 \
      --max-val 64 --output runs/probe-adapter --adapter-name probe --no-mlflow 2>&1 | tee logs/probe.log
    ;;
  train)
    log="logs/train_${ADAPTER}_$(date +%Y%m%d-%H%M).log"
    tmux new-session -d -s t2s-train \
      "$PY -m src.finetuning.train_lora --epochs $EPOCHS --output adapters/$ADAPTER 2>&1 | tee $log"
    echo "Training started in tmux session 't2s-train' -> $log"
    echo "Watch: tmux attach -t t2s-train   (detach: Ctrl-b d)   |   tail -f $log"
    ;;
  eval)
    "$PY" -m src.serving.hf_server --adapter "$ADAPTER=adapters/$ADAPTER" --port 8001 > logs/hf_server.log 2>&1 &
    server=$!
    trap 'kill $server' EXIT
    until curl -sf localhost:8001/health >/dev/null; do sleep 5; done
    T2S_LLM_BASE_URL=http://localhost:8001/v1 "$PY" -m src.evaluation.compare_approaches \
      --split dev --adapter "$ADAPTER" --concurrency 1 ${EVAL_ARGS:-} 2>&1 | tee logs/eval.log
    ;;
  *)
    sed -n '2,13p' "$0"; exit 2 ;;
esac
