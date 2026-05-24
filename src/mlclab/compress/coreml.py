from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mlclab.config import CompressionKind, CompressionRecipe


def _copy_artifact(source: Path, destination: Path) -> Path:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    return destination


def compress_coreml_artifact(
    source: Path,
    destination: Path,
    recipe: CompressionRecipe,
    calibration_data: Iterable[dict[str, Any]] | None = None,
) -> Path:
    if recipe.kind == CompressionKind.FP16:
        return _copy_artifact(source, destination)

    try:
        import coremltools as ct
        import coremltools.optimize as cto
    except ImportError as exc:
        raise RuntimeError(
            "Core ML compression requires coremltools. Install the ml extra."
        ) from exc

    mlmodel = ct.models.MLModel(str(source))
    if recipe.kind == CompressionKind.INT8_WEIGHT_ONLY:
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode=recipe.mode or "linear_symmetric",
                dtype="int8",
            )
        )
        compressed = cto.coreml.linear_quantize_weights(mlmodel, config)
    elif recipe.kind == CompressionKind.PALETTIZED:
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode=recipe.mode or "kmeans",
                nbits=recipe.bits,
            )
        )
        compressed = cto.coreml.palettize_weights(mlmodel, config)
    elif recipe.kind == CompressionKind.PRUNED_SPARSE:
        config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpMagnitudePrunerConfig(
                target_sparsity=recipe.target_sparsity,
            )
        )
        compressed = cto.coreml.prune_weights(mlmodel, config)
    else:
        raise ValueError(f"unsupported compression recipe: {recipe.kind}")

    if recipe.calibration_samples is not None:
        if calibration_data is None:
            raise ValueError("calibration_data is required when calibration_samples is set")
        sample_data = list(calibration_data)
        if not sample_data:
            raise ValueError("calibration_data must contain at least one sample")
        activation_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode=recipe.mode or "linear_symmetric",
            )
        )
        compressed = cto.coreml.linear_quantize_activations(
            compressed,
            activation_config,
            sample_data,
            calibration_op_group_size=recipe.calibration_op_group_size,
        )

    compressed.save(str(destination))
    return destination
