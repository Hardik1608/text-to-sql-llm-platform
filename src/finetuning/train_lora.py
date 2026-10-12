"""
QLoRA Fine-Tuning Script for Text-to-SQL (sized for a single 12 GB GPU, e.g. RTX 3060).

  base model : Qwen/Qwen2.5-3B-Instruct, frozen, loaded in 4-bit NF4 + double quantization
  adapter    : LoRA r=16, alpha=32, dropout 0.05 on all attention + MLP projections (~30M params)
  objective  : SFT with loss on the SQL completion only (schema/question tokens are masked)
  tracking   : MLflow — params, loss curves, eval loss, peak VRAM, adapter artifact
  output     : adapters/<name>/  (vLLM-loadable LoRA) + adapter_manifest.json

Input is produced by `python -m src.finetuning.prepare_data` (data/finetune/{train,val}.jsonl).

Usage:
    python -m src.finetuning.train_lora --output adapters/spider-v1
    python -m src.finetuning.train_lora --epochs 2 --batch-size 2 --grad-accum 8   # if OOM
    python -m src.finetuning.train_lora --smoke-test   # CPU, tiny model, 2 steps: checks the code path
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.generation.prompts import PROMPT_VERSION

log = logging.getLogger("train_lora")

QWEN2_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SMOKE_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

# Arguments renamed across TRL / transformers releases: our name -> alternatives to try.
# (transformers 5 folded warmup_ratio into warmup_steps: a float < 1 is read as a ratio.)
_RENAMED = {
    "max_length": "max_seq_length",
    "eval_strategy": "evaluation_strategy",
    "warmup_ratio": "warmup_steps",
}
DEFAULT_MLFLOW_URI = "sqlite:///mlflow.db"  # MLflow 3.x no longer accepts the ./mlruns file store


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def make_sft_config(**kwargs) -> SFTConfig:
    """Build an SFTConfig, mapping argument names across TRL versions."""
    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    final = {}
    for key, value in kwargs.items():
        if key in accepted:
            final[key] = value
        elif _RENAMED.get(key) in accepted:
            final[_RENAMED[key]] = value
        else:
            log.warning("SFTConfig in this TRL version has no '%s'; ignoring it", key)
    return SFTConfig(**final)


def precision() -> str:
    if not torch.cuda.is_available():
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"  # RTX 30xx (Ampere): bf16


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(args: argparse.Namespace, prec: str):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[prec]
    if args.quant == "4bit":
        if not torch.cuda.is_available():
            raise SystemExit("4-bit QLoRA needs a CUDA GPU (use --quant none for CPU runs)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat4 — information-optimal for normal weights
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,  # quantize the quantization constants (~0.4 GB saved)
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map={"": 0}, torch_dtype=dtype
        )
        # Casts norms to fp32, enables input grads -> required for gradient checkpointing on k-bit
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        if torch.cuda.is_available():
            model = model.to("cuda")
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.rank,  # rank: 8/16/32/64 — capacity vs. trainable params
        lora_alpha=args.alpha,  # scaling = alpha / r (here 2.0)
        lora_dropout=args.dropout,
        bias="none",
        target_modules=QWEN2_TARGET_MODULES,  # all attention + MLP projections
    )
    model = get_peft_model(model, lora_config)
    return model, lora_config


def load_splits(data_dir: Path, max_train: int | None, max_val: int | None):
    files = {"train": str(data_dir / "train.jsonl"), "validation": str(data_dir / "val.jsonl")}
    ds = load_dataset("json", data_files=files)
    ds = ds.select_columns(["prompt", "completion"])  # conversational prompt-completion format
    train, val = ds["train"], ds["validation"]
    if max_train:
        train = train.select(range(min(max_train, len(train))))
    if max_val:
        val = val.select(range(min(max_val, len(val))))
    return train, val


# -----------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------


def train(args: argparse.Namespace) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.smoke_test:
        args.model, args.quant, args.max_steps = args.smoke_model, "none", 2
        args.max_train, args.max_val, args.batch_size, args.grad_accum = 16, 8, 2, 1
        args.eval_steps = args.save_steps = 1
        args.gradient_checkpointing = False

    adapter_dir = Path(args.output)
    adapter_name = args.adapter_name or adapter_dir.name
    run_dir = Path(args.run_dir) / adapter_name
    prec = precision()
    log.info(
        "precision=%s cuda=%s model=%s quant=%s",
        prec,
        torch.cuda.is_available(),
        args.model,
        args.quant,
    )

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # 2. Base model (4-bit) + LoRA adapter
    # ------------------------------------------------------------------
    model, lora_config = load_model(args, prec)
    trainable, total = model.get_nb_trainable_parameters()
    log.info(
        "trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    # ------------------------------------------------------------------
    # 3. Data (built by prepare_data.py — same prompts as inference)
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    train_ds, val_ds = load_splits(data_dir, args.max_train, args.max_val)
    log.info("train=%d val=%d", len(train_ds), len(val_ds))

    # ------------------------------------------------------------------
    # 4. Training configuration
    # ------------------------------------------------------------------
    use_mlflow = not args.no_mlflow
    sft_config = make_sft_config(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if args.quant == "4bit" else "adamw_torch",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=prec == "bf16",
        fp16=prec == "fp16",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        # Keep the LAST weights. Measured on the RTX 3060 run: val loss was lowest at step 100
        # (0.204) and rose to 0.401 by epoch 3, yet execution accuracy rose from 67.5% to 75.5%
        # (qlora, 200 dev questions). Loss punishes confident answers; it is the wrong selector.
        load_best_model_at_end=False,
        max_length=args.max_seq_len,
        packing=False,
        completion_only_loss=True,  # loss on the SQL tokens only
        report_to=["mlflow"] if use_mlflow else "none",
        dataloader_num_workers=0,  # Windows-friendly
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 5. SFT Trainer
    # ------------------------------------------------------------------
    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
    }
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:  # TRL < 0.12
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    # ------------------------------------------------------------------
    # 6. Train with MLflow tracking
    # ------------------------------------------------------------------
    stats_path = data_dir / "stats.json"
    params = {
        "base_model": args.model,
        "adapter_name": adapter_name,
        "lora_rank": args.rank,
        "lora_alpha": args.alpha,
        "lora_dropout": args.dropout,
        "target_modules": ",".join(QWEN2_TARGET_MODULES),
        "quantization": "qlora-nf4-double" if args.quant == "4bit" else "none",
        "precision": prec,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "train_data_sha256": file_sha256(data_dir / "train.jsonl")[:16],
        "prompt_version": PROMPT_VERSION,
    }
    run = None
    if use_mlflow:
        import mlflow

        # Shared by our logging and the HF Trainer's MLflow callback
        os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(args.experiment)
        run = mlflow.start_run(
            run_name=args.run_name or f"qlora-r{args.rank}-a{args.alpha}-{adapter_name}"
        )
        mlflow.log_params(params)
        if stats_path.exists():
            mlflow.log_artifact(str(stats_path), artifact_path="data")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    try:
        trainer.train(resume_from_checkpoint=args.resume)
        eval_metrics = trainer.evaluate()
        runtime_s = time.time() - start
        peak_vram_gb = (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        )
        log.info("eval=%s runtime=%.0fs peak_vram=%.2f GB", eval_metrics, runtime_s, peak_vram_gb)

        # ------------------------------------------------------------------
        # 7. Save the adapter (vLLM-loadable) + manifest
        # ------------------------------------------------------------------
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        manifest = {
            "adapter_name": adapter_name,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **params,
            "eval_loss": eval_metrics.get("eval_loss"),
            "train_runtime_s": round(runtime_s, 1),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "global_steps": trainer.state.global_step,
            "mlflow_run_id": run.info.run_id if run else None,
            "data_stats": json.loads(stats_path.read_text()) if stats_path.exists() else None,
        }
        (adapter_dir / "adapter_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if run is not None:
            import mlflow

            mlflow.log_metrics(
                {"final_eval_loss": eval_metrics.get("eval_loss", float("nan")),
                 "peak_vram_gb": peak_vram_gb, "train_runtime_s": runtime_s}
            )  # fmt: skip
            mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")
            mlflow.set_tags({"adapter_name": adapter_name, "stage": "candidate"})
        log.info("Adapter saved to %s", adapter_dir)
    finally:
        if run is not None:
            import mlflow

            mlflow.end_run()
    return adapter_dir


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for Text-to-SQL")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument(
        "--max-steps", type=int, default=0, dest="max_steps", help="override epochs (benchmarks)"
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    p.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    p.add_argument("--max-seq-len", type=int, default=1536, dest="max_seq_len")
    p.add_argument(
        "--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing"
    )
    p.add_argument("--logging-steps", type=int, default=10, dest="logging_steps")
    p.add_argument("--eval-steps", type=int, default=100, dest="eval_steps")
    p.add_argument("--save-steps", type=int, default=100, dest="save_steps")
    p.add_argument("--max-train", type=int, default=None, dest="max_train")
    p.add_argument("--max-val", type=int, default=None, dest="max_val")
    p.add_argument("--data-dir", default="data/finetune")
    p.add_argument("--output", default="adapters/spider-v1", help="final adapter directory")
    p.add_argument("--adapter-name", default=None, help="defaults to the output dir name")
    p.add_argument("--run-dir", default="runs", help="checkpoints go to <run-dir>/<adapter-name>")
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--experiment", default="text-to-sql-qlora")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true", help="tiny model, CPU, 2 steps")
    p.add_argument("--smoke-model", default=SMOKE_MODEL)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
