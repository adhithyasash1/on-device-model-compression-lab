from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from mlclab.artifacts import RunContext, RunStore
from mlclab.benchmark.memory import rss_bytes
from mlclab.config import CompressionKind, ComputeUnit, RunConfig
from mlclab.metrics import classification_summary, latency_summary_ms, top_k_indices
from mlclab.metrics.classification import top5_agreement
from mlclab.reports.scoreboard import regenerate_scoreboard


class RunStageError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def _split_limit(split: str, fallback: int) -> int:
    digits = "".join(char for char in split if char.isdigit())
    return int(digits) if digits else fallback


def plan_run(config: RunConfig, repo_root: Path) -> dict[str, Any]:
    return {
        "repo_root": str(repo_root),
        "run_name": config.name,
        "execution_mode": config.execution.mode,
        "model": config.model.model_dump(mode="json"),
        "dataset": config.dataset.model_dump(mode="json"),
        "export": config.export.model_dump(mode="json"),
        "compression": config.compression.model_dump(mode="json"),
        "recovery": config.recovery.model_dump(mode="json"),
        "benchmark": config.benchmark.model_dump(mode="json"),
        "output_dir": str(repo_root / config.output_dir),
        "reports_dir": str(repo_root / config.reports_dir),
    }


def run_config(config: RunConfig, repo_root: Path) -> dict[str, Any]:
    store = RunStore(repo_root / config.output_dir)
    context = store.create(config, repo_root)
    try:
        if config.execution.mode == "synthetic":
            summary = _run_synthetic(config, context, store)
        else:
            summary = _run_real(config, context, store, repo_root)
    except RunStageError as exc:
        summary = _failure_summary(config, context, exc.stage, exc)
        store.write_summary(context, summary)
        store.log_event(context, "run_failed", {"stage": exc.stage, "error": str(exc)})
    except Exception as exc:
        summary = _failure_summary(config, context, "benchmark_failed", exc)
        store.write_summary(context, summary)
        store.log_event(context, "run_failed", {"stage": "benchmark_failed", "error": str(exc)})

    regenerate_scoreboard(repo_root / config.output_dir, repo_root / config.reports_dir)
    return summary


def _synthetic_logits(
    label: int, num_labels: int, rng: random.Random, *, drift: bool
) -> list[float]:
    logits = [rng.random() * 0.02 for _ in range(num_labels)]
    logits[label] = 1.0
    if drift and label + 1 < num_labels:
        logits[label + 1] = 1.01
    return logits


def _run_synthetic(config: RunConfig, context: RunContext, store: RunStore) -> dict[str, Any]:
    rng = random.Random(config.dataset.seed)
    sample_count = config.dataset.limit or _split_limit(config.dataset.split, 20)
    artifact_path = context.artifacts_dir / f"{config.compression.name}.mlpackage"
    artifact_path.mkdir()
    (artifact_path / "SYNTHETIC_ARTIFACT.txt").write_text(
        "Synthetic artifact for testing mlclab machinery.\n",
        encoding="utf-8",
    )

    memory = {"before_load_rss_bytes": rss_bytes()}
    records: list[dict[str, Any]] = []
    for index in range(sample_count):
        label = index % config.model.num_labels
        pytorch_logits = _synthetic_logits(label, config.model.num_labels, rng, drift=False)
        artifact_logits = list(pytorch_logits)
        pytorch_top5 = top_k_indices(pytorch_logits, 5)
        artifact_top5 = top_k_indices(artifact_logits, 5)
        record = {
            "type": "prediction",
            "sample_id": f"synthetic_{index:05d}",
            "label": label,
            "pytorch_top5": pytorch_top5,
            "artifact_top5": artifact_top5,
            "top1_correct": artifact_top5[0] == label,
            "top5_correct": label in artifact_top5,
            "top5_agreement": top5_agreement(pytorch_top5, artifact_top5),
        }
        records.append(record)
        store.append_metric(context, record)

    latency: dict[str, dict[str, float | int]] = {}
    for compute_unit in config.benchmark.compute_units:
        durations = _synthetic_latency_ns(compute_unit, config.benchmark.iterations)
        latency[compute_unit.value] = latency_summary_ms(durations)
        for iteration, duration in enumerate(durations):
            store.append_metric(context, _latency_record(compute_unit.value, iteration, duration))

    memory["after_benchmark_rss_bytes"] = rss_bytes()
    summary = _base_summary(config, context, "ok")
    summary.update(
        {
            "artifact": {
                "path": str(artifact_path),
                "size_bytes": _artifact_size(artifact_path),
            },
            "accuracy": classification_summary(records),
            "latency": latency,
            "memory": memory,
        }
    )
    store.write_summary(context, summary)
    store.log_event(context, "run_completed", {"status": "ok"})
    return summary


def _synthetic_latency_ns(compute_unit: ComputeUnit, iterations: int) -> list[int]:
    base_ms = {
        ComputeUnit.CPU_ONLY: 9.0,
        ComputeUnit.CPU_AND_GPU: 5.0,
        ComputeUnit.CPU_AND_NE: 4.0,
        ComputeUnit.ALL: 3.5,
    }[compute_unit]
    rng = random.Random(f"{compute_unit.value}:{iterations}")
    durations = []
    for _ in range(iterations):
        jitter = rng.uniform(-0.15, 0.15)
        durations.append(int((base_ms + jitter) * 1_000_000))
    return durations


def _run_real(
    config: RunConfig,
    context: RunContext,
    store: RunStore,
    repo_root: Path,
) -> dict[str, Any]:
    from mlclab.benchmark.coreml_runner import (
        load_coreml_model,
        measure_predictions,
        predict_logits,
    )
    from mlclab.compress import compress_coreml_artifact
    from mlclab.export import export_coreml_fp16
    from mlclab.models import load_mobilevit
    from mlclab.recovery import top_k_with_scores

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("real runs require torch. Install the ml extra.") from exc

    if config.dataset.name != "imagenette":
        raise NotImplementedError("real v1 currently supports the imagenette dataset only")
    if config.dataset.root is None:
        raise ValueError("real imagenette runs require dataset.root")

    limit = config.dataset.limit or _split_limit(config.dataset.split, 20)
    _, _, manifest = _load_imagenette_records(
        config,
        repo_root,
        limit=limit,
        seed=config.dataset.seed,
        use_manifest_path=True,
    )
    if not manifest:
        raise ValueError("dataset manifest is empty")

    store.log_event(context, "loading_model", {"model": config.model.id})
    memory = {"before_load_rss_bytes": rss_bytes()}
    bundle = load_mobilevit(config.model.id, config.model.revision)
    memory["after_model_load_rss_bytes"] = rss_bytes()

    example_input = torch.rand(1, 3, config.model.input_size, config.model.input_size)
    fp16_path = context.artifacts_dir / "coreml_fp16.mlpackage"
    try:
        export_coreml_fp16(
            bundle.wrapper,
            example_input,
            fp16_path,
            input_name=bundle.input_name,
            deployment_target=config.export.deployment_target,
            precision=config.export.precision,
        )
    except Exception as exc:
        raise RunStageError("conversion_failed", f"Core ML export failed: {exc}") from exc

    artifact_path = fp16_path
    eval_sample_ids = {sample["sample_id"] for sample in manifest}
    compression_calibration = {
        "samples_requested": config.compression.calibration_samples,
        "samples_used": 0,
        "status": "disabled" if config.compression.calibration_samples is None else "not_run",
    }
    if config.compression.kind != CompressionKind.FP16:
        artifact_path = context.artifacts_dir / f"{config.compression.name}.mlpackage"
        calibration_data = _compression_calibration_data(
            config,
            repo_root,
            bundle,
            excluded_sample_ids=eval_sample_ids,
        )
        if config.compression.calibration_samples is not None:
            compression_calibration.update(
                {
                    "samples_used": len(calibration_data or []),
                    "status": "ok",
                    "calibration_seed": config.compression.calibration_seed
                    if config.compression.calibration_seed is not None
                    else config.dataset.seed + 1,
                    "excluded_eval_samples": len(eval_sample_ids),
                }
            )
        try:
            compress_coreml_artifact(
                fp16_path,
                artifact_path,
                config.compression,
                calibration_data=calibration_data,
            )
        except Exception as exc:
            raise RunStageError(
                "conversion_failed",
                f"Core ML compression failed for {config.compression.name}: {exc}",
            ) from exc

    # Decode, preprocessing, accuracy probes, and tensor materialization happen before timing.
    tensors: list[Any] = []
    records: list[dict[str, Any]] = []
    mlmodel = load_coreml_model(artifact_path, config.benchmark.compute_units[0].value)
    repair, recovery = _fit_logit_repair(
        config,
        context,
        repo_root,
        bundle,
        mlmodel,
        excluded_sample_ids=eval_sample_ids,
    )
    for sample in manifest:
        pixel_values = _sample_pixel_values(bundle, sample)
        with torch.no_grad():
            pytorch_logits = bundle.wrapper(pixel_values)
        tensor = pixel_values.detach().cpu().numpy()
        artifact_logits_raw = predict_logits(mlmodel, bundle.input_name, tensor)
        artifact_logits = (
            repair.apply(artifact_logits_raw) if repair is not None else artifact_logits_raw
        )
        tensors.append(tensor)

        pytorch_top5 = top_k_indices(pytorch_logits, 5)
        artifact_top5 = top_k_indices(artifact_logits, 5)
        record = {
            "type": "prediction",
            "sample_id": sample["sample_id"],
            "label": sample["label"],
            "pytorch_top5": pytorch_top5,
            "artifact_top5": artifact_top5,
            "top1_correct": artifact_top5[0] == sample["label"],
            "top5_correct": sample["label"] in artifact_top5,
            "top5_agreement": top5_agreement(pytorch_top5, artifact_top5),
        }
        if repair is not None:
            record.update(
                {
                    "artifact_top5_raw": top_k_indices(artifact_logits_raw, 5),
                    "artifact_logits_raw_top5": top_k_with_scores(artifact_logits_raw, 5),
                    "logit_repair": "class_bias",
                }
            )
        records.append(record)
        store.append_metric(context, record)

    latency: dict[str, dict[str, float | int]] = {}
    for compute_unit in config.benchmark.compute_units:
        # Model loading, potential Core ML compilation, and warmup are outside timed samples.
        lane_model = load_coreml_model(artifact_path, compute_unit.value)
        durations = measure_predictions(
            lane_model,
            bundle.input_name,
            tensors,
            warmup=config.benchmark.warmup,
            iterations=config.benchmark.iterations,
        )
        latency[compute_unit.value] = latency_summary_ms(durations)
        for iteration, duration in enumerate(durations):
            store.append_metric(context, _latency_record(compute_unit.value, iteration, duration))

    memory["after_benchmark_rss_bytes"] = rss_bytes()
    summary = _base_summary(config, context, "ok")
    summary.update(
        {
            "artifact": {
                "path": str(artifact_path),
                "size_bytes": _artifact_size(artifact_path),
            },
            "accuracy": classification_summary(records),
            "latency": latency,
            "memory": memory,
            "compression_calibration": compression_calibration,
            "recovery": recovery,
        }
    )
    store.write_summary(context, summary)
    store.log_event(context, "run_completed", {"status": "ok"})
    return summary


def _load_imagenette_records(
    config: RunConfig,
    repo_root: Path,
    *,
    limit: int | None,
    seed: int,
    use_manifest_path: bool,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    from mlclab.data import (
        build_imagenette_manifest,
        load_imagenette_manifest,
        resolve_imagenette160_roots,
    )

    if config.dataset.root is None:
        raise ValueError("imagenette records require dataset.root")

    dataset_root, val_root = resolve_imagenette160_roots(
        _resolve_repo_path(repo_root, config.dataset.root)
    )
    if use_manifest_path and config.dataset.manifest_path is not None:
        records = load_imagenette_manifest(
            _resolve_repo_path(repo_root, config.dataset.manifest_path),
            dataset_root=dataset_root,
        )
        return dataset_root, val_root, records[:limit] if limit is not None else records

    return (
        dataset_root,
        val_root,
        build_imagenette_manifest(val_root, limit=limit, seed=seed),
    )


def _sample_pixel_values(bundle: Any, sample: dict[str, Any]) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("image preprocessing requires Pillow. Install the ml extra.") from exc

    image = Image.open(sample["path"]).convert("RGB")
    inputs = bundle.processor(images=image, return_tensors="pt")
    return inputs[bundle.input_name]


def _compression_calibration_data(
    config: RunConfig,
    repo_root: Path,
    bundle: Any,
    *,
    excluded_sample_ids: set[str],
) -> list[dict[str, Any]] | None:
    if config.compression.calibration_samples is None:
        return None

    seed = (
        config.compression.calibration_seed
        if config.compression.calibration_seed is not None
        else config.dataset.seed + 1
    )
    records = _calibration_records(
        config,
        repo_root,
        limit=config.compression.calibration_samples,
        seed=seed,
        excluded_sample_ids=excluded_sample_ids,
    )
    return [
        {bundle.input_name: _sample_pixel_values(bundle, sample).detach().cpu().numpy()}
        for sample in records
    ]


def _fit_logit_repair(
    config: RunConfig,
    context: RunContext,
    repo_root: Path,
    bundle: Any,
    mlmodel: Any,
    *,
    excluded_sample_ids: set[str],
) -> tuple[Any | None, dict[str, Any]]:
    if config.recovery.logit_repair == "none":
        return (
            None,
            {
                "logit_repair": "none",
                "status": "disabled",
                "calibration_samples_requested": None,
                "calibration_samples_used": 0,
            },
        )

    if config.recovery.logit_repair != "class_bias":
        raise ValueError(f"unsupported logit repair: {config.recovery.logit_repair}")
    if config.recovery.calibration_samples is None:
        raise ValueError("class-bias repair requires recovery.calibration_samples")

    seed = (
        config.recovery.calibration_seed
        if config.recovery.calibration_seed is not None
        else config.dataset.seed + 2
    )
    records = _calibration_records(
        config,
        repo_root,
        limit=config.recovery.calibration_samples,
        seed=seed,
        excluded_sample_ids=excluded_sample_ids,
    )
    repair = _fit_class_bias_repair(bundle, mlmodel, records)
    repair_path = context.artifacts_dir / "class_bias_repair.json"
    repair.write_json(repair_path)

    repair_summary = {key: value for key, value in repair.to_dict().items() if key != "bias"}
    repair_summary.update(
        {
            "status": "ok",
            "logit_repair": "class_bias",
            "calibration_samples_requested": config.recovery.calibration_samples,
            "calibration_samples_used": len(records),
            "calibration_seed": seed,
            "artifact_path": str(repair_path),
        }
    )
    return repair, repair_summary


def _calibration_records(
    config: RunConfig,
    repo_root: Path,
    *,
    limit: int,
    seed: int,
    excluded_sample_ids: set[str],
) -> list[dict[str, Any]]:
    _, _, records = _load_imagenette_records(
        config,
        repo_root,
        limit=None,
        seed=seed,
        use_manifest_path=False,
    )
    filtered = [sample for sample in records if str(sample["sample_id"]) not in excluded_sample_ids]
    return filtered[:limit]


def _fit_class_bias_repair(bundle: Any, mlmodel: Any, records: list[dict[str, Any]]) -> Any:
    from mlclab.benchmark.coreml_runner import predict_logits
    from mlclab.recovery import ClassBiasRepair

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("class-bias repair requires torch. Install the ml extra.") from exc

    pairs = []
    for sample in records:
        pixel_values = _sample_pixel_values(bundle, sample)
        with torch.no_grad():
            pytorch_logits = bundle.wrapper(pixel_values)
        artifact_logits = predict_logits(
            mlmodel,
            bundle.input_name,
            pixel_values.detach().cpu().numpy(),
        )
        pairs.append((pytorch_logits, artifact_logits))
    return ClassBiasRepair.fit(pairs)


def _base_summary(config: RunConfig, context: RunContext, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "execution_mode": config.execution.mode,
        "run_dir": str(context.root),
        "name": config.name,
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "dataset": {
            "name": config.dataset.name,
            "split": config.dataset.split,
            "seed": config.dataset.seed,
            "limit": config.dataset.limit,
        },
        "recipe": config.compression.model_dump(mode="json"),
        "recovery": {
            "logit_repair": config.recovery.logit_repair,
            "calibration_samples": config.recovery.calibration_samples,
            "calibration_seed": config.recovery.calibration_seed,
            "status": "disabled" if config.recovery.logit_repair == "none" else "not_run",
        },
        "export": config.export.model_dump(mode="json"),
        "created_at_unix": time.time(),
    }


def _latency_record(compute_unit: str, iteration: int, duration_ns: int) -> dict[str, Any]:
    return {
        "type": "latency",
        "compute_unit": compute_unit,
        "iteration": iteration,
        "duration_ns": duration_ns,
    }


def _failure_summary(
    config: RunConfig,
    context: RunContext,
    status: str,
    exc: Exception,
) -> dict[str, Any]:
    summary = _base_summary(config, context, status)
    summary.update(
        {
            "failure_stage": status,
            "failure_summary": str(exc),
            "artifact": {"path": None, "size_bytes": 0},
            "accuracy": {
                "sample_count": 0,
                "top1": 0.0,
                "top5": 0.0,
                "top5_agreement": 0.0,
            },
            "latency": {},
            "memory": {},
        }
    )
    (context.logs_dir / "failure.json").write_text(
        json.dumps({"error": str(exc), "type": type(exc).__name__}, indent=2),
        encoding="utf-8",
    )
    return summary


def _artifact_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path
