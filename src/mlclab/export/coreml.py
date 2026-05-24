from __future__ import annotations

from pathlib import Path
from typing import Any


def export_coreml_fp16(
    torch_model: Any,
    example_input: Any,
    output_path: Path,
    *,
    input_name: str,
    deployment_target: str,
    precision: str,
) -> Path:
    try:
        import coremltools as ct
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Core ML export requires torch and coremltools. Install the ml extra."
        ) from exc

    torch_model.eval()
    with torch.no_grad():
        if hasattr(torch, "export"):
            exported_model = torch.export.export(torch_model, (example_input,))
            traced = exported_model.run_decompositions({})
        else:
            traced = torch.jit.trace(torch_model, example_input)

    target = getattr(ct.target, deployment_target, None)
    if target is None:
        raise ValueError(f"unknown Core ML deployment target: {deployment_target}")

    compute_precision = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=input_name, shape=tuple(example_input.shape))],
        minimum_deployment_target=target,
        compute_precision=compute_precision,
    )
    mlmodel.save(str(output_path))
    return output_path
