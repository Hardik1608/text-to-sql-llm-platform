"""
4-way benchmark: prompting vs schema RAG vs QLoRA vs QLoRA + RAG on Spider.

Every example runs through the production pipeline (retrieval → generation → safety layer →
execution), so the numbers describe the system that serves traffic. Reports execution
accuracy, exact match, SQL validity, hallucination rate, latency, tokens and cost, overall
and per Spider hardness level.

Usage:
    python -m src.evaluation.compare_approaches --split dev
    python -m src.evaluation.compare_approaches --split dev --modes prompt,rag --limit 200
    python -m src.evaluation.compare_approaches --split dev --adapter spider-v1 --mlflow

Outputs (reports/<timestamp>_<split>/):
    predictions_<mode>.jsonl   per-example predictions + metrics
    summary.json               all metrics, all modes
    summary.md                 README-ready tables
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from datetime import datetime
from pathlib import Path

from src.config import Settings
from src.evaluation.metrics import (
    HARDNESS_LEVELS,
    EvalRecord,
    classify_hardness,
    exact_match,
    order_matters,
    results_match,
    summarize,
)
from src.factory import build_pipeline
from src.generation.prompts import ADAPTER_MODES, MODES, PROMPT_VERSION
from src.pipeline import QueryRequest, Text2SQLPipeline
from src.safety.executor import ExecutionError, SafeSQLiteExecutor

MODE_LABELS = {
    "prompt": "Base + Prompting",
    "rag": "Base + Schema RAG",
    "qlora": "QLoRA Fine-Tuned",
    "qlora_rag": "QLoRA + Schema RAG",
}


def load_split(
    processed_dir: Path, split: str, limit: int | None, sample: bool, seed: int
) -> list[dict]:
    with open(processed_dir / f"{split}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for i, row in enumerate(rows):
        row["idx"] = i
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit) if sample else rows[:limit]
    return rows


def execute_gold(
    pipeline: Text2SQLPipeline, rows: list[dict], timeout_s: float
) -> dict[int, list | None]:
    executor = SafeSQLiteExecutor(timeout_s=timeout_s, max_rows=None)
    gold: dict[int, list | None] = {}
    for row in rows:
        db_path = pipeline.catalog.db_path(row["db_id"])
        try:
            gold[row["idx"]] = executor.execute(db_path, row["query"]).rows if db_path else None
        except ExecutionError:
            gold[row["idx"]] = None  # a handful of Spider gold queries fail on their own DBs
    return gold


async def run_mode(
    pipeline: Text2SQLPipeline,
    mode: str,
    rows: list[dict],
    gold: dict[int, list | None],
    adapter: str | None,
    concurrency: int,
) -> list[EvalRecord]:
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def one(row: dict) -> EvalRecord:
        nonlocal done
        async with semaphore:
            result = await pipeline.run(
                QueryRequest(
                    question=row["question"],
                    db_id=row["db_id"],
                    mode=mode,
                    adapter=adapter if mode in ADAPTER_MODES else None,
                    execute=True,
                    max_rows=None,
                    use_cache=False,
                )
            )
        gold_rows = gold[row["idx"]]
        match = (
            result.status == "ok"
            and gold_rows is not None
            and results_match(result.rows or [], gold_rows, order_matters(row["query"]))
        )
        done += 1
        if done % 50 == 0 or done == len(rows):
            print(f"  [{mode}] {done}/{len(rows)}", flush=True)
        return EvalRecord(
            idx=row["idx"],
            db_id=row["db_id"],
            question=row["question"],
            gold_sql=row["query"],
            mode=mode,
            pred_sql=result.sql,
            status=result.status,
            exec_match=match,
            exact_match=exact_match(result.sql, row["query"], pipeline.catalog.get(row["db_id"])),
            hardness=classify_hardness(row["query"]),
            latency_ms=result.timings.get("total_ms", 0.0),
            generation_ms=result.timings.get("generation_ms", 0.0),
            prompt_tokens=result.usage["prompt_tokens"],
            completion_tokens=result.usage["completion_tokens"],
            violation_codes=[v["code"] for v in result.violations],
            error=result.error,
            repaired=result.repaired,
        )

    return list(await asyncio.gather(*(one(r) for r in rows)))


def render_markdown(summary: dict, meta: dict) -> str:
    lines = [
        f"# Benchmark - Spider {meta['split']} (n={meta['n']})",
        "",
        f"- Base model: `{meta['base_model']}`; adapter: `{meta['adapter'] or '-'}`; "
        f"prompt version `{meta['prompt_version']}`; repairs: {meta['max_repairs']}",
        f"- LLM backend: `{meta['llm_base_url']}`; run at {meta['timestamp']}",
        "",
        "| Approach | Exec Acc | Exact Match | Valid SQL | Hallucination | P50 latency | P95 latency | Prompt tok | Cost / 1k q |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, s in summary.items():
        lines.append(
            f"| {MODE_LABELS[mode]} | {s['execution_accuracy']:.1%} | {s['exact_match']:.1%} | "
            f"{s['valid_sql_rate']:.1%} | {s['hallucination_rate']:.1%} | {s['latency_ms_p50']:.0f} ms | "
            f"{s['latency_ms_p95']:.0f} ms | {s['prompt_tokens_mean']:.0f} | ${s['cost_per_1k_queries_usd']:.3f} |"
        )
    lines += ["", "Execution accuracy by Spider hardness:", ""]
    lines.append("| Approach | " + " | ".join(h.capitalize() for h in HARDNESS_LEVELS) + " |")
    lines.append("|---" * (len(HARDNESS_LEVELS) + 1) + "|")
    for mode, s in summary.items():
        cells = []
        for h in HARDNESS_LEVELS:
            b = s["by_hardness"].get(h)
            cells.append(f"{b['execution_accuracy']:.1%} (n={b['n']})" if b else "-")
        lines.append(f"| {MODE_LABELS[mode]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def log_to_mlflow(summary: dict, meta: dict, out_dir: Path) -> None:
    import os

    import mlflow  # optional dependency (requirements-train.txt)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("text-to-sql-eval")
    with mlflow.start_run(run_name=f"eval-{meta['split']}-{meta['timestamp']}"):
        mlflow.log_params({k: v for k, v in meta.items() if not isinstance(v, list | dict)})
        for mode, s in summary.items():
            mlflow.log_metrics(
                {f"{mode}.{k}": v for k, v in s.items() if isinstance(v, int | float)}
            )
        mlflow.log_artifacts(str(out_dir))


async def main(args: argparse.Namespace) -> Path:
    updates = {
        "cache_enabled": False,
        "rag_num_examples": args.num_examples,
        "rag_max_tables": args.max_tables,
        "rag_value_hints": not args.no_value_hints,
    }
    if args.llm_base_url:
        updates["llm_base_url"] = args.llm_base_url
    if args.base_model:
        updates["base_model"] = args.base_model
    settings = Settings().model_copy(update=updates)

    pipeline = build_pipeline(settings, cache=None)
    pipeline.max_repair_attempts = args.max_repairs
    pipeline.executor = SafeSQLiteExecutor(timeout_s=args.timeout_s, max_rows=None)
    if args.adapter and args.adapter not in pipeline.registry:
        pipeline.registry.register(args.adapter)
    adapter = args.adapter or pipeline.registry.default

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - set(MODES)
    if unknown:
        raise SystemExit(f"Unknown modes: {sorted(unknown)}")
    if adapter is None and ADAPTER_MODES & set(modes):
        print("No LoRA adapter configured (--adapter or T2S_ADAPTERS): skipping qlora modes")
        modes = [m for m in modes if m not in ADAPTER_MODES]

    rows = load_split(settings.processed_dir, args.split, args.limit, args.sample, args.seed)
    print(f"Evaluating {len(rows)} {args.split} examples in modes {modes}")
    gold = execute_gold(pipeline, rows, args.timeout_s)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / f"{timestamp}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for mode in modes:
        start = time.perf_counter()
        records = await run_mode(pipeline, mode, rows, gold, adapter, args.concurrency)
        wall_s = time.perf_counter() - start
        with open(out_dir / f"predictions_{mode}.jsonl", "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        s = summarize(records)
        s["wall_time_s"] = round(wall_s, 1)
        s["throughput_qps"] = round(len(records) / wall_s, 3) if wall_s else 0.0
        s["cost_per_1k_queries_usd"] = round(
            args.gpu_cost_per_hour * wall_s / 3600 / len(records) * 1000, 4
        )
        summary[mode] = s
        print(f"  [{mode}] EX={s['execution_accuracy']:.1%} EM={s['exact_match']:.1%} "
              f"valid={s['valid_sql_rate']:.1%} p95={s['latency_ms_p95']:.0f}ms")  # fmt: skip

    meta = {
        "split": args.split,
        "n": len(rows),
        "modes": modes,
        "base_model": settings.base_model,
        "adapter": adapter if ADAPTER_MODES & set(modes) else None,
        "llm_base_url": settings.llm_base_url,
        "prompt_version": PROMPT_VERSION,
        "max_repairs": args.max_repairs,
        "num_examples": args.num_examples,
        "max_tables": args.max_tables,
        "value_hints": not args.no_value_hints,
        "concurrency": args.concurrency,
        "gpu_cost_per_hour": args.gpu_cost_per_hour,
        "gold_exec_errors": sum(v is None for v in gold.values()),
        "timestamp": timestamp,
    }
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "results": summary}, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(render_markdown(summary, meta), encoding="utf-8")
    (Path(args.output_dir) / "LATEST").write_text(out_dir.name, encoding="utf-8")
    print("\n" + render_markdown(summary, meta))
    print(f"Report written to {out_dir}")

    if args.mlflow:
        log_to_mlflow(summary, meta, out_dir)
    await pipeline.llm.aclose()
    return out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare prompting / RAG / QLoRA / QLoRA+RAG on Spider")
    p.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--limit", type=int, default=None, help="evaluate only N examples")
    p.add_argument("--sample", action="store_true", help="random N examples instead of the first N")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--adapter", default=None, help="LoRA adapter name for the qlora modes")
    p.add_argument("--max-repairs", type=int, default=0, help="0 = measure the raw model")
    p.add_argument("--num-examples", type=int, default=3, help="retrieved few-shot examples (RAG)")
    p.add_argument("--max-tables", type=int, default=6, help="schema pruning budget (RAG)")
    p.add_argument("--no-value-hints", action="store_true")
    p.add_argument("--timeout-s", type=float, default=30.0, help="per-query execution timeout")
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--base-model", default=None)
    p.add_argument("--gpu-cost-per-hour", type=float, default=0.0, help="USD/h of the serving GPU")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--mlflow", action="store_true", help="log metrics + report to MLflow")
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
