"""
Prepare Spider data for supervised fine-tuning (SFT).

Builds TRL conversational prompt-completion examples with the SAME ContextBuilder and prompt
templates the API uses, so the adapter is trained on exactly the prompts it will see:

  --context full   every example uses the full schema          (serving mode `qlora`)
  --context rag    every example uses retrieved context        (serving mode `qlora_rag`)
  --context mixed  each example randomly gets one of the two   (default: one adapter, both modes)

Leakage controls:
  - retrieved few-shot examples never come from the example's own database — at evaluation
    time they always come from *other* databases, so training must match that
  - pruned schemas always keep the gold tables (no training on unanswerable prompts)
  - the validation split is held out BY DATABASE (Spider is cross-domain); Spider dev is never
    used for training or checkpoint selection

Usage:
    python -m src.finetuning.prepare_data
    python -m src.finetuning.prepare_data --context full --max-len 1536
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from src.config import Settings
from src.generation.prompts import format_sft_example
from src.retrieval.context import ContextBuilder
from src.retrieval.example_retriever import ExampleRetriever
from src.retrieval.schema_retriever import SchemaRetriever
from src.retrieval.value_index import ValueIndex
from src.safety.sql_validator import SQLValidator
from src.schema import SchemaCatalog

PROCESSED_DIR = Path("data/processed")
FINETUNE_DIR = Path("data/finetune")


class TokenCounter:
    """Exact counts with the model's chat template when transformers is available."""

    def __init__(self, model_name: str | None):
        self.tokenizer = None
        if model_name:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:  # transformers missing / offline
                print(f"Tokenizer unavailable ({type(e).__name__}); estimating tokens as chars/3.2")

    def __call__(self, example: dict) -> int:
        messages = example["prompt"] + example["completion"]
        if self.tokenizer is not None:
            # return_dict=True: transformers 5 returns a dict by default (len() would be 2)
            encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)
            return len(encoded["input_ids"])
        return math.ceil(sum(len(m["content"]) for m in messages) / 3.2)


def split_by_database(records: list[dict], val_fraction: float, seed: int) -> tuple[list, list]:
    db_ids = sorted({r["db_id"] for r in records})
    random.Random(seed).shuffle(db_ids)
    val_dbs = set(db_ids[: max(1, round(len(db_ids) * val_fraction))])
    train = [r for r in records if r["db_id"] not in val_dbs]
    val = [r for r in records if r["db_id"] in val_dbs]
    return train, val


def build_examples(
    records: list[dict],
    builder: ContextBuilder,
    validator: SQLValidator,
    context: str,
    rag_ratio: float,
    rng: random.Random,
) -> list[dict]:
    out = []
    for rec in records:
        use_rag = context == "rag" or (context == "mixed" and rng.random() < rag_ratio)
        mode = "qlora_rag" if use_rag else "qlora"
        schema = builder.catalog.get(rec["db_id"])
        gold_tables = validator.validate(rec["query"], schema).tables
        ctx = builder.build(
            mode,
            rec["question"],
            rec["db_id"],
            exclude_example_db=rec["db_id"],
            must_include_tables=gold_tables,
        )
        ex = format_sft_example(
            rec, mode=mode, schema_context=ctx.schema_sql,
            examples=ctx.examples, value_hints=ctx.value_hints,
        )  # fmt: skip
        ex["context"] = "rag" if use_rag else "full"
        out.append(ex)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            digest.update(line.encode("utf-8"))
            f.write(line)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> dict:
    settings = Settings()
    catalog = SchemaCatalog.from_json(args.processed_dir / "schemas.json", settings.db_dirs)
    with open(args.processed_dir / "train.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if args.max_examples:
        records = random.Random(args.seed).sample(records, min(args.max_examples, len(records)))

    train_recs, val_recs = split_by_database(records, args.val_fraction, args.seed)
    # Few-shot pool = training split only: validation SQL must never appear in training prompts
    builder = ContextBuilder(
        catalog=catalog,
        schema_retriever=SchemaRetriever.from_catalog(catalog),
        example_retriever=ExampleRetriever.from_records(train_recs, catalog),
        value_index=ValueIndex(catalog) if not args.no_value_hints else None,
        num_examples=args.num_examples,
        max_tables=args.max_tables,
    )
    validator = SQLValidator(check_columns=False)
    count_tokens = TokenCounter(None if args.no_tokenizer else args.model)

    stats: dict = {"context": args.context, "max_len": args.max_len, "splits": {}}
    rng = random.Random(args.seed)
    for split, recs in (("train", train_recs), ("val", val_recs)):
        examples = build_examples(recs, builder, validator, args.context, args.rag_ratio, rng)
        kept, lengths = [], []
        for ex in examples:
            ex["n_tokens"] = count_tokens(ex)
            if ex["n_tokens"] <= args.max_len:
                kept.append(ex)
                lengths.append(ex["n_tokens"])
        sha = write_jsonl(args.output_dir / f"{split}.jsonl", kept)
        stats["splits"][split] = {
            "examples": len(kept),
            "skipped_too_long": len(examples) - len(kept),
            "databases": len({ex["db_id"] for ex in kept}),
            "rag_context": sum(ex["context"] == "rag" for ex in kept),
            "tokens_mean": round(statistics.fmean(lengths), 1) if lengths else 0,
            "tokens_p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))] if lengths else 0,
            "tokens_max": max(lengths, default=0),
            "tokens_total": sum(lengths),
            "sha256": sha,
        }
        s = stats["splits"][split]
        print(
            f"[{split}] written={s['examples']} skipped(too long)={s['skipped_too_long']} "
            f"dbs={s['databases']} rag={s['rag_context']} tokens mean={s['tokens_mean']} "
            f"p95={s['tokens_p95']} max={s['tokens_max']}"
        )
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Output: {args.output_dir}")
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SFT data for Text-to-SQL QLoRA")
    p.add_argument("--context", default="mixed", choices=["full", "rag", "mixed"])
    p.add_argument(
        "--rag-ratio", type=float, default=0.5, help="share of RAG prompts in mixed mode"
    )
    p.add_argument("--val-fraction", type=float, default=0.05, help="share of DATABASES held out")
    p.add_argument(
        "--model", default="Qwen/Qwen2.5-3B-Instruct", help="tokenizer for length filtering"
    )
    p.add_argument(
        "--no-tokenizer", action="store_true", help="estimate token counts (no transformers)"
    )
    p.add_argument("--max-len", type=int, default=1536, dest="max_len")
    p.add_argument("--num-examples", type=int, default=3)
    p.add_argument("--max-tables", type=int, default=6)
    p.add_argument("--no-value-hints", action="store_true")
    p.add_argument("--max-examples", type=int, default=None, help="subsample (quick experiments)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--output-dir", type=Path, default=FINETUNE_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    prepare(parse_args())
