import json
from pathlib import Path
from typing import Any

import pytest

from mlclab.benchmark.coreml_runner import measure_predictions
from mlclab.config import ComputeUnit, load_config
from mlclab.metrics import latency_summary_ms
from mlclab.pipeline.run import run_config


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def predict(self, inputs: dict[str, Any]) -> dict[str, list[float]]:
        self.calls.append(inputs["pixel_values"])
        return {"logits": [1.0]}


def test_measure_predictions_times_only_prepared_coreml_predict_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecordingModel()
    ticks = iter([100, 160, 200, 275, 300, 390])
    time_calls: list[int] = []

    def fake_perf_counter_ns() -> int:
        value = next(ticks)
        time_calls.append(value)
        return value

    monkeypatch.setattr(
        "mlclab.benchmark.coreml_runner.time.perf_counter_ns",
        fake_perf_counter_ns,
    )

    durations = measure_predictions(
        model,
        "pixel_values",
        ["tensor-a", "tensor-b"],
        warmup=2,
        iterations=3,
    )

    assert durations == [60, 75, 90]
    assert time_calls == [100, 160, 200, 275, 300, 390]
    assert model.calls == [
        "tensor-a",
        "tensor-b",
        "tensor-a",
        "tensor-b",
        "tensor-a",
    ]


def test_measure_predictions_requires_warmup_to_exclude_lazy_compilation() -> None:
    with pytest.raises(ValueError, match="warmup prediction"):
        measure_predictions(
            RecordingModel(),
            "pixel_values",
            ["tensor-a"],
            warmup=0,
            iterations=1,
        )


def test_synthetic_benchmark_latency_record_structure(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke/synthetic_fp16.yaml"))
    config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(update={"limit": 3, "split": "eval3"}),
            "benchmark": config.benchmark.model_copy(
                update={
                    "compute_units": [ComputeUnit.CPU_ONLY],
                    "iterations": 4,
                    "warmup": 1,
                }
            ),
        }
    )

    summary = run_config(config, tmp_path)
    metrics_path = Path(summary["run_dir"]) / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]

    prediction_records = [record for record in records if record["type"] == "prediction"]
    latency_records = [record for record in records if record["type"] == "latency"]

    assert len(prediction_records) == 3
    assert len(latency_records) == 4
    assert all(
        set(record)
        == {"type", "sample_id", "label", "pytorch_top5", "artifact_top5"}
        | {"top1_correct", "top5_correct", "top5_agreement"}
        for record in prediction_records
    )
    assert all(
        set(record) == {"type", "compute_unit", "iteration", "duration_ns"}
        for record in latency_records
    )
    assert [record["iteration"] for record in latency_records] == [0, 1, 2, 3]
    assert all(record["compute_unit"] == "CPU_ONLY" for record in latency_records)
    assert all(isinstance(record["duration_ns"], int) for record in latency_records)
    assert all(record["duration_ns"] > 0 for record in latency_records)
    assert summary["latency"]["CPU_ONLY"] == latency_summary_ms(
        [record["duration_ns"] for record in latency_records]
    )
