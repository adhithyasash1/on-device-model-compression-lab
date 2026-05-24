from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def logits_to_vector(logits: Any) -> np.ndarray:
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    array = np.asarray(logits, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("logits are empty")
    return array


def tensor_stats(tensor: Any) -> dict[str, Any]:
    array = np.asarray(tensor)
    values = array.astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def top_k_with_scores(logits: Any, k: int = 5) -> list[dict[str, float | int]]:
    vector = logits_to_vector(logits)
    k = min(k, vector.size)
    indices = np.argsort(vector)[::-1][:k]
    return [
        {"rank": rank, "class_id": int(index), "logit": float(vector[index])}
        for rank, index in enumerate(indices, start=1)
    ]


@dataclass(frozen=True)
class ClassBiasRepair:
    bias: np.ndarray
    sample_count: int

    @classmethod
    def fit(cls, pairs: list[tuple[Any, Any]]) -> ClassBiasRepair:
        if not pairs:
            raise ValueError("class-bias repair requires at least one calibration pair")

        deltas = []
        expected_size: int | None = None
        for pytorch_logits, artifact_logits in pairs:
            pytorch_vector = logits_to_vector(pytorch_logits)
            artifact_vector = logits_to_vector(artifact_logits)
            if pytorch_vector.shape != artifact_vector.shape:
                raise ValueError(
                    "pytorch and artifact logits must have the same shape for class-bias repair"
                )
            if expected_size is None:
                expected_size = pytorch_vector.size
            elif pytorch_vector.size != expected_size:
                raise ValueError("all calibration logits must have the same size")
            deltas.append(pytorch_vector - artifact_vector)

        return cls(bias=np.mean(np.stack(deltas, axis=0), axis=0), sample_count=len(pairs))

    def apply(self, logits: Any) -> np.ndarray:
        vector = logits_to_vector(logits)
        if vector.shape != self.bias.shape:
            raise ValueError("logit vector shape does not match class-bias repair shape")
        return vector + self.bias

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "class_bias",
            "sample_count": self.sample_count,
            "num_classes": int(self.bias.size),
            "mean_abs_bias": float(np.mean(np.abs(self.bias))),
            "max_abs_bias": float(np.max(np.abs(self.bias))),
            "bias": [float(value) for value in self.bias],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassBiasRepair:
        if data.get("method") != "class_bias":
            raise ValueError(f"unsupported repair method: {data.get('method')}")
        return cls(
            bias=np.asarray(data["bias"], dtype=np.float64),
            sample_count=int(data["sample_count"]),
        )

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path) -> ClassBiasRepair:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
