from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import mlclab.cli
from mlclab.cli import app
from mlclab.data import IMAGENETTE_LABELS


def test_cli_validate() -> None:
    result = CliRunner().invoke(app, ["validate", "configs/smoke/synthetic_fp16.yaml"])

    assert result.exit_code == 0
    assert "ok: synthetic-fp16-smoke" in result.output


def test_cli_plan() -> None:
    result = CliRunner().invoke(app, ["plan", "configs/smoke/synthetic_fp16.yaml"])

    assert result.exit_code == 0
    assert "synthetic-fp16-smoke" in result.output


def test_cli_prepare_imagenette160(tmp_path: Path) -> None:
    dataset_root = tmp_path / "imagenette2-160"
    manifests_dir = tmp_path / "manifests"
    for synset in IMAGENETTE_LABELS:
        folder = dataset_root / "val" / synset
        folder.mkdir(parents=True)
        for index in range(50):
            (folder / f"{synset}_{index:04d}.JPEG").write_bytes(b"image")

    result = CliRunner().invoke(
        app,
        [
            "prepare-imagenette160",
            "--root",
            str(dataset_root),
            "--manifests-dir",
            str(manifests_dir),
            "--no-download",
        ],
    )

    assert result.exit_code == 0
    assert "eval20:" in result.output
    assert "eval500:" in result.output
    assert (manifests_dir / "imagenette2-160-eval20.jsonl").is_file()
    assert (manifests_dir / "imagenette2-160-eval500.metadata.json").is_file()


def test_cli_run_strict_failure_exits_nonzero(monkeypatch: Any, tmp_path: Path) -> None:
    def failed_run(config: Any, repo_root: Path) -> dict[str, str]:
        return {"status": "benchmark_failed", "run_dir": str(repo_root / "runs" / "failed")}

    monkeypatch.setattr(mlclab.cli, "run_config", failed_run)

    result = CliRunner().invoke(
        app,
        ["run", "configs/smoke/synthetic_fp16.yaml", "--repo-root", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "status: benchmark_failed" in result.output


def test_cli_debug_sample_prints_top5(monkeypatch: Any, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "debug-run"
    run_dir.mkdir(parents=True)

    def fake_debug_sample(
        run_dir_arg: Path,
        sample_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert run_dir_arg == run_dir
        assert sample_id == "sample-1"
        assert kwargs["dump_logits"] is True
        return {
            "run_dir": str(run_dir),
            "artifact_path": str(run_dir / "artifacts" / "model.mlpackage"),
            "compute_unit": "CPU_ONLY",
            "sample_id": "sample-1",
            "label": 217,
            "synset": "n02102040",
            "tensor_stats": {
                "shape": [1, 3, 256, 256],
                "dtype": "float32",
                "min": -1.0,
                "max": 1.0,
                "mean": 0.0,
                "std": 0.5,
            },
            "pytorch_top5": [{"rank": 1, "class_id": 217, "logit": 9.0}],
            "coreml_top5": [{"rank": 1, "class_id": 217, "logit": 8.5}],
            "logits_path": str(run_dir / "logs" / "debug_sample-1_logits.json"),
        }

    monkeypatch.setattr(mlclab.cli, "debug_sample", fake_debug_sample)

    result = CliRunner().invoke(
        app,
        [
            "debug",
            "sample",
            "--run",
            str(run_dir),
            "--sample-id",
            "sample-1",
            "--dump-logits",
        ],
    )

    assert result.exit_code == 0
    assert "label: 217 (n02102040)" in result.output
    assert "tensor: shape=[1, 3, 256, 256]" in result.output
    assert "PyTorch top-5:" in result.output
    assert "Core ML top-5:" in result.output
    assert "logits:" in result.output
