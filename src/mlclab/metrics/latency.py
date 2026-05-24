from __future__ import annotations

import statistics


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def latency_summary_ms(durations_ns: list[int]) -> dict[str, float | int]:
    durations_ms = sorted(value / 1_000_000 for value in durations_ns)
    if not durations_ms:
        return {
            "count": 0,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "mean_ms": 0.0,
            "stddev_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(durations_ms),
        "median_ms": statistics.median(durations_ms),
        "p90_ms": percentile(durations_ms, 0.90),
        "p95_ms": percentile(durations_ms, 0.95),
        "mean_ms": statistics.fmean(durations_ms),
        "stddev_ms": statistics.pstdev(durations_ms),
        "min_ms": durations_ms[0],
        "max_ms": durations_ms[-1],
    }
