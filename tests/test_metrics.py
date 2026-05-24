import pytest

from mlclab.metrics import classification_summary, latency_summary_ms, top_k_indices


def test_top_k_indices() -> None:
    assert top_k_indices([0.1, 0.8, 0.2], 2) == [1, 2]


def test_classification_summary() -> None:
    summary = classification_summary(
        [
            {"top1_correct": True, "top5_correct": True, "top5_agreement": True},
            {"top1_correct": False, "top5_correct": True, "top5_agreement": False},
        ]
    )

    assert summary["sample_count"] == 2
    assert summary["top1"] == 0.5
    assert summary["top5"] == 1.0
    assert summary["top5_agreement"] == 0.5


def test_latency_summary_ms() -> None:
    summary = latency_summary_ms([1_000_000, 2_000_000, 3_000_000])

    assert summary["count"] == 3
    assert summary["median_ms"] == 2.0
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 3.0


def test_latency_summary_ms_percentiles_mean_and_population_stddev() -> None:
    summary = latency_summary_ms([1_000_000, 2_000_000, 4_000_000, 8_000_000])

    assert summary["count"] == 4
    assert summary["median_ms"] == 3.0
    assert summary["p90_ms"] == pytest.approx(6.8)
    assert summary["p95_ms"] == pytest.approx(7.4)
    assert summary["mean_ms"] == 3.75
    assert summary["stddev_ms"] == pytest.approx(2.68095132369)
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 8.0


def test_latency_summary_ms_empty_input() -> None:
    summary = latency_summary_ms([])

    assert summary == {
        "count": 0,
        "median_ms": 0.0,
        "p90_ms": 0.0,
        "p95_ms": 0.0,
        "mean_ms": 0.0,
        "stddev_ms": 0.0,
        "min_ms": 0.0,
        "max_ms": 0.0,
    }
