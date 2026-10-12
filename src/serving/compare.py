"""
Side-by-side comparison: one question through every approach at once.

For Spider's held-out questions (dev / test) the gold SQL is known, so each approach is graded
exactly as the benchmark grades it (execution match: same rows, any row order unless the gold
query sorts, any column order). For other questions there is no ground truth; approaches are
grouped by identical results instead ("3 of 4 agree").
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluation.metrics import order_matters, results_match
from src.generation.prompts import MODES
from src.pipeline import QueryRequest, Text2SQLPipeline
from src.safety.executor import ExecutionError
from src.serving.vllm_client import AdapterNotFoundError

HELD_OUT_SPLITS = ("dev", "test")  # never used for training or prompts' few-shot pool
DISPLAY_ROWS = 50


def _normalize(question: str) -> str:
    return " ".join(question.lower().split()).rstrip(" ?.!")


@dataclass
class BenchmarkQuestion:
    question: str
    gold_sql: str
    split: str


@dataclass
class BenchmarkIndex:
    """Spider held-out questions with their gold SQL, keyed by (db_id, normalized question)."""

    by_key: dict[tuple[str, str], BenchmarkQuestion] = field(default_factory=dict)
    by_db: dict[str, list[BenchmarkQuestion]] = field(default_factory=dict)

    @classmethod
    def load(cls, processed_dir: Path) -> BenchmarkIndex:
        index = cls()
        for split in HELD_OUT_SPLITS:
            path = processed_dir / f"{split}.jsonl"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    item = BenchmarkQuestion(row["question"], row["query"], split)
                    key = (row["db_id"], _normalize(row["question"]))
                    if key not in index.by_key:
                        index.by_key[key] = item
                        index.by_db.setdefault(row["db_id"], []).append(item)
        return index

    def lookup(self, db_id: str, question: str) -> BenchmarkQuestion | None:
        return self.by_key.get((db_id, _normalize(question)))

    def questions(self, db_id: str, limit: int) -> list[BenchmarkQuestion]:
        return self.by_db.get(db_id, [])[:limit]


def _trim(result: dict) -> dict:
    rows = result.get("rows")
    if rows is not None and len(rows) > DISPLAY_ROWS:
        result["rows"] = rows[:DISPLAY_ROWS]
        result["truncated"] = True
    return result


def _signature(rows: list[list] | None) -> tuple | None:
    """Order-insensitive fingerprint of a result set, for grouping approaches that agree."""
    if rows is None:
        return None
    normalized = [tuple(round(v, 6) if isinstance(v, float) else v for v in row) for row in rows]
    return tuple(sorted(Counter(map(repr, normalized)).items()))


async def compare_modes(
    pipeline: Text2SQLPipeline,
    index: BenchmarkIndex,
    question: str,
    db_id: str,
    modes: tuple[str, ...] = MODES,
    use_cache: bool = False,
) -> dict:
    """Run `question` through every mode concurrently and grade or group the results."""

    async def one(mode: str) -> dict:
        try:
            result = await pipeline.run(
                QueryRequest(
                    question=question, db_id=db_id, mode=mode, max_rows=None, use_cache=use_cache
                )  # fmt: skip
            )
        except AdapterNotFoundError as e:
            return {"mode": mode, "status": "unavailable", "error": str(e.args[0])}
        return result.to_dict()

    results = list(await asyncio.gather(*(one(m) for m in modes)))

    gold = None
    bench = index.lookup(db_id, question)
    gold_rows = None
    if bench is not None:
        db_path = pipeline.catalog.db_path(db_id)
        try:
            out = await asyncio.to_thread(pipeline.executor.execute, db_path, bench.gold_sql, None)
            gold_rows = [list(r) for r in out.rows]
            gold = _trim({"sql": bench.gold_sql, "split": bench.split, "columns": out.columns,
                          "rows": gold_rows, "row_count": out.row_count})  # fmt: skip
        except (ExecutionError, TypeError):
            gold = {"sql": bench.gold_sql, "split": bench.split, "error": "gold query failed"}

    signatures = {}
    for r in results:
        ok = r.get("status") == "ok"
        if gold_rows is not None:
            r["correct"] = bool(
                ok and results_match(r.get("rows") or [], gold_rows, order_matters(bench.gold_sql))
            )
        else:
            r["correct"] = None
        signatures[r["mode"]] = _signature(r.get("rows")) if ok else None

    groups: dict[tuple, list[str]] = {}
    for mode, sig in signatures.items():
        if sig is not None:
            groups.setdefault(sig, []).append(mode)
    agreement = sorted(groups.values(), key=len, reverse=True)
    for r in results:
        r["agrees_with"] = next((g for g in agreement if r["mode"] in g), [])
        _trim(r)

    return {
        "question": question,
        "db_id": db_id,
        "gold": gold,
        "results": results,
        "agreement": agreement,
        "correct_count": sum(1 for r in results if r.get("correct"))
        if gold_rows is not None
        else None,
    }
