import json
from pathlib import Path

from mlclab.reports.scoreboard import regenerate_scoreboard


def test_regenerate_scoreboard(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "execution_mode": "synthetic",
                "run_dir": "/stale/copied/workspace/run1",
                "model_id": "synthetic/model",
                "dataset": {"name": "synthetic", "split": "eval20"},
                "recipe": {"name": "fp16"},
                "artifact": {"size_bytes": 2_000_000},
                "accuracy": {
                    "sample_count": 20,
                    "top1": 1.0,
                    "top5": 1.0,
                    "top5_agreement": 1.0,
                },
                "latency": {"ALL": {"median_ms": 3.1}},
                "memory": {
                    "before_load_rss_bytes": 100_000_000,
                    "after_benchmark_rss_bytes": 125_000_000,
                },
                "failure_stage": "",
            }
        ),
        encoding="utf-8",
    )

    rows = regenerate_scoreboard(tmp_path / "runs", tmp_path / "reports")

    assert rows[0]["recipe"] == "fp16"
    assert rows[0]["mode"] == "synthetic"
    assert rows[0]["run_id"] == "run1"
    assert rows[0]["size_mb"] == 2.0
    assert rows[0]["rss_delta_mb"] == 25.0
    assert rows[0]["rss_after_benchmark_mb"] == 125.0
    assert (tmp_path / "reports" / "scoreboard.csv").exists()
    assert (tmp_path / "reports" / "scoreboard.md").exists()


def test_real_benchmark_table_keeps_latest_real_row(tmp_path: Path) -> None:
    older_dir = tmp_path / "runs" / "older"
    newer_dir = tmp_path / "runs" / "newer"
    older_dir.mkdir(parents=True)
    newer_dir.mkdir(parents=True)
    base_summary = {
        "status": "ok",
        "execution_mode": "real",
        "model_id": "apple/mobilevit-xx-small",
        "dataset": {"name": "imagenette", "split": "eval20"},
        "recipe": {"name": "fp16"},
        "artifact": {"size_bytes": 2_000_000},
        "accuracy": {"sample_count": 20, "top1": 0.5, "top5": 0.9, "top5_agreement": 1.0},
        "latency": {"ALL": {"median_ms": 3.1}},
        "memory": {
            "before_load_rss_bytes": 100_000_000,
            "after_benchmark_rss_bytes": 140_000_000,
        },
    }
    (older_dir / "summary.json").write_text(
        json.dumps(base_summary | {"created_at_unix": 1.0}),
        encoding="utf-8",
    )
    (newer_dir / "summary.json").write_text(
        json.dumps(base_summary | {"created_at_unix": 2.0, "accuracy": {"top1": 0.7}}),
        encoding="utf-8",
    )

    regenerate_scoreboard(tmp_path / "runs", tmp_path / "reports")

    table = (tmp_path / "reports" / "real_benchmark_table.md").read_text(encoding="utf-8")
    assert "| newer |" in table
    assert str(newer_dir.resolve()) not in table
    assert str(older_dir.resolve()) not in table


def test_scoreboard_includes_failure_stage(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "conversion_failed",
                "execution_mode": "real",
                "failure_stage": "conversion_failed",
                "failure_summary": "Core ML compression failed",
                "model_id": "apple/mobilevit-xx-small",
                "dataset": {"name": "imagenette", "split": "eval20"},
                "recipe": {"name": "palettized-6bit"},
                "artifact": {"size_bytes": 0},
                "accuracy": {"sample_count": 0},
                "latency": {},
            }
        ),
        encoding="utf-8",
    )

    rows = regenerate_scoreboard(tmp_path / "runs", tmp_path / "reports")

    assert rows[0]["failure_stage"] == "conversion_failed"
    assert rows[0]["failure_summary"] == "Core ML compression failed"
