from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mlclab.benchmark.coreml_runner import load_coreml_model, predict_logits
from mlclab.config import RunConfig, load_config
from mlclab.pipeline.run import _load_imagenette_records, _sample_pixel_values
from mlclab.recovery import ClassBiasRepair, tensor_stats, top_k_with_scores


def debug_sample(
    run_dir: Path,
    sample_id: str,
    *,
    repo_root: Path | None = None,
    compute_unit: str | None = None,
    dump_logits: bool = False,
    logits_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    config = load_config(run_dir / "config.yaml")
    if config.execution.mode != "real":
        raise ValueError("debug sample requires a real run with a Core ML artifact")
    if config.dataset.name != "imagenette":
        raise NotImplementedError("debug sample currently supports imagenette runs")

    resolved_repo_root = _infer_repo_root(run_dir, config, repo_root)
    artifact_path = _artifact_path(run_dir, resolved_repo_root)
    unit = compute_unit or config.benchmark.compute_units[0].value
    sample = _find_sample(config, resolved_repo_root, sample_id)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("debug sample requires torch. Install the ml extra.") from exc

    from mlclab.models import load_mobilevit

    bundle = load_mobilevit(config.model.id, config.model.revision)
    pixel_values = _sample_pixel_values(bundle, sample)
    with torch.no_grad():
        pytorch_logits = bundle.wrapper(pixel_values)

    tensor = pixel_values.detach().cpu().numpy()
    mlmodel = load_coreml_model(artifact_path, unit)
    coreml_logits = predict_logits(mlmodel, bundle.input_name, tensor)
    repair = _load_repair(run_dir)
    repaired_logits = repair.apply(coreml_logits) if repair is not None else None

    result = {
        "run_dir": str(run_dir),
        "artifact_path": str(artifact_path),
        "compute_unit": unit,
        "sample_id": sample["sample_id"],
        "label": sample["label"],
        "synset": sample.get("synset"),
        "image_path": sample["path"],
        "tensor_stats": tensor_stats(tensor),
        "pytorch_top5": top_k_with_scores(pytorch_logits, 5),
        "coreml_top5": top_k_with_scores(coreml_logits, 5),
    }
    if repaired_logits is not None:
        result["coreml_repaired_top5"] = top_k_with_scores(repaired_logits, 5)

    if dump_logits or logits_path is not None:
        output_path = logits_path or run_dir / "logs" / f"debug_{_safe_slug(sample_id)}_logits.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = result | {
            "pytorch_logits": _logits_list(pytorch_logits),
            "coreml_logits": _logits_list(coreml_logits),
        }
        if repaired_logits is not None:
            output["coreml_repaired_logits"] = _logits_list(repaired_logits)
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        result["logits_path"] = str(output_path)

    return result


def _find_sample(config: RunConfig, repo_root: Path, sample_id: str) -> dict[str, Any]:
    limit = config.dataset.limit or _split_limit(config.dataset.split, 20)
    _, _, records = _load_imagenette_records(
        config,
        repo_root,
        limit=limit,
        seed=config.dataset.seed,
        use_manifest_path=True,
    )
    for sample in records:
        if sample["sample_id"] == sample_id:
            return sample
    raise ValueError(f"sample_id not found in run manifest: {sample_id}")


def _artifact_path(run_dir: Path, repo_root: Path) -> Path:
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        raw_artifact_path = summary.get("artifact", {}).get("path")
        if raw_artifact_path:
            path = _resolve_path(Path(raw_artifact_path), run_dir, repo_root)
            if path.exists():
                return path

    candidates = sorted((run_dir / "artifacts").glob("*.mlpackage"))
    if not candidates:
        raise FileNotFoundError(f"no Core ML artifact found under {run_dir / 'artifacts'}")
    return candidates[-1]


def _load_repair(run_dir: Path) -> ClassBiasRepair | None:
    repair_path = run_dir / "artifacts" / "class_bias_repair.json"
    if not repair_path.is_file():
        return None
    return ClassBiasRepair.read_json(repair_path)


def _infer_repo_root(run_dir: Path, config: RunConfig, repo_root: Path | None) -> Path:
    if repo_root is not None:
        return repo_root.resolve()
    if not config.output_dir.is_absolute() and run_dir.parent.name == config.output_dir.name:
        return run_dir.parent.parent.resolve()
    return Path.cwd().resolve()


def _resolve_path(path: Path, run_dir: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    for base in (run_dir, repo_root):
        candidate = base / path
        if candidate.exists():
            return candidate
    return repo_root / path


def _logits_list(logits: Any) -> list[float]:
    from mlclab.recovery import logits_to_vector

    return [float(value) for value in logits_to_vector(logits)]


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower() or "sample"


def _split_limit(split: str, fallback: int) -> int:
    digits = "".join(char for char in split if char.isdigit())
    return int(digits) if digits else fallback
