from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def load_coreml_model(artifact_path: Path, compute_unit: str) -> Any:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError(
            "Core ML benchmarking requires coremltools. Install the ml extra."
        ) from exc

    try:
        unit = getattr(ct.ComputeUnit, compute_unit)
    except AttributeError as exc:
        raise ValueError(f"unknown Core ML compute unit: {compute_unit}") from exc
    return ct.models.MLModel(str(artifact_path), compute_units=unit)


def predict_logits(mlmodel: Any, input_name: str, tensor: Any) -> Any:
    prediction = mlmodel.predict({input_name: tensor})
    if not prediction:
        raise RuntimeError("Core ML prediction returned no outputs")
    return next(iter(prediction.values()))


def measure_predictions(
    mlmodel: Any,
    input_name: str,
    tensors: list[Any],
    *,
    warmup: int,
    iterations: int,
) -> list[int]:
    """Measure only Core ML prediction calls against already prepared tensors."""
    if not tensors:
        raise ValueError("at least one tensor is required for latency measurement")
    if warmup < 1:
        raise ValueError("at least one warmup prediction is required before timing")

    for index in range(warmup):
        predict_logits(mlmodel, input_name, tensors[index % len(tensors)])

    durations: list[int] = []
    for index in range(iterations):
        tensor = tensors[index % len(tensors)]
        start = time.perf_counter_ns()
        predict_logits(mlmodel, input_name, tensor)
        durations.append(time.perf_counter_ns() - start)
    return durations
