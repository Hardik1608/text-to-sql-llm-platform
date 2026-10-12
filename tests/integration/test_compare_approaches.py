import asyncio
import json

from src.evaluation import compare_approaches
from src.factory import build_pipeline as real_build_pipeline
from tests.conftest import DEV, FakeLLM


def test_benchmark_end_to_end(data_dir, tmp_path, monkeypatch):
    # Gold SQL for 3 of 4 dev questions; the 4th gets a wrong (but valid) query
    script = {q: [s] for _, q, s in DEV[:3]}
    llm = FakeLLM(script=script, default="SELECT count(*) FROM stadium")
    monkeypatch.setenv("T2S_DATA_DIR", str(data_dir))
    monkeypatch.setattr(
        compare_approaches,
        "build_pipeline",
        lambda settings, cache=None: real_build_pipeline(
            settings, llm=llm.client(max_retries=0), cache=cache
        ),
    )
    args = compare_approaches.parse_args(
        ["--split", "dev", "--modes", "prompt,rag,qlora", "--adapter", "spider-v1",
         "--output-dir", str(tmp_path), "--gpu-cost-per-hour", "0.5"]
    )  # fmt: skip
    out_dir = asyncio.run(compare_approaches.main(args))

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["meta"]["n"] == 4 and summary["meta"]["adapter"] == "spider-v1"
    for mode in ("prompt", "rag", "qlora"):
        result = summary["results"][mode]
        assert result["execution_accuracy"] == 0.75
        assert result["valid_sql_rate"] == 1.0
        assert result["cost_per_1k_queries_usd"] >= 0
    preds = [json.loads(line) for line in (out_dir / "predictions_rag.jsonl").open()]
    assert len(preds) == 4 and sum(p["exec_match"] for p in preds) == 3
    assert "QLoRA Fine-Tuned" in (out_dir / "summary.md").read_text()
    assert (tmp_path / "LATEST").read_text() == out_dir.name
