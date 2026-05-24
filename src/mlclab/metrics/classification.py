from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def _as_scores(logits: Any) -> list[float]:
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    if hasattr(logits, "reshape") and hasattr(logits, "tolist"):
        logits = logits.reshape(-1).tolist()
    return [float(value) for value in logits]


def top_k_indices(logits: Any, k: int = 5) -> list[int]:
    scores = _as_scores(logits)
    k = min(k, len(scores))
    return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]


def is_top_k_correct(logits: Any, label: int, k: int = 5) -> bool:
    return label in top_k_indices(logits, k)


def classification_summary(records: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    total = 0
    top1 = 0
    top5 = 0
    top5_agreement = 0
    for record in records:
        total += 1
        top1 += int(record.get("top1_correct", False))
        top5 += int(record.get("top5_correct", False))
        top5_agreement += int(record.get("top5_agreement", False))

    if total == 0:
        return {
            "sample_count": 0,
            "top1": 0.0,
            "top5": 0.0,
            "top5_agreement": 0.0,
        }

    return {
        "sample_count": total,
        "top1": top1 / total,
        "top5": top5 / total,
        "top5_agreement": top5_agreement / total,
    }


def top5_agreement(left: Sequence[int], right: Sequence[int]) -> bool:
    return list(left[:5]) == list(right[:5])
