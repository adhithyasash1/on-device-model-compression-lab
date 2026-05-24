from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

SCOREBOARD_COLUMNS = [
    "status",
    "mode",
    "failure_stage",
    "model_id",
    "recipe",
    "dataset",
    "split",
    "samples",
    "size_mb",
    "top1",
    "top5",
    "top5_agreement",
    "best_median_ms",
    "best_compute_unit",
    "rss_delta_mb",
    "rss_after_benchmark_mb",
    "failure_summary",
    "run_id",
]

REAL_BENCHMARK_COLUMNS = [
    "status",
    "model_id",
    "recipe",
    "split",
    "samples",
    "size_mb",
    "top1",
    "top5",
    "top5_agreement",
    "best_median_ms",
    "best_compute_unit",
    "rss_delta_mb",
    "rss_after_benchmark_mb",
    "tradeoff",
    "failure_stage",
    "failure_summary",
    "run_id",
]


def load_run_summaries(runs_root: Path) -> list[dict[str, Any]]:
    if not runs_root.exists():
        return []
    summaries = []
    for summary_path in sorted(runs_root.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        summary["run_id"] = summary_path.parent.name
        summaries.append(summary)
    return summaries


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    latency = summary.get("latency", {})
    best_unit = ""
    best_median = ""
    if latency:
        best_unit, best_stats = min(
            latency.items(),
            key=lambda item: item[1].get("median_ms", float("inf")),
        )
        best_median = best_stats.get("median_ms", "")

    artifact = summary.get("artifact", {})
    accuracy = summary.get("accuracy", {})
    dataset = summary.get("dataset", {})
    recipe = summary.get("recipe", {})
    memory = summary.get("memory", {})
    size_bytes = artifact.get("size_bytes") or 0
    before_load_rss = memory.get("before_load_rss_bytes")
    after_benchmark_rss = memory.get("after_benchmark_rss_bytes")
    return {
        "status": summary.get("status", ""),
        "mode": summary.get("execution_mode", ""),
        "failure_stage": summary.get("failure_stage", ""),
        "model_id": summary.get("model_id", ""),
        "recipe": recipe.get("name", ""),
        "dataset": dataset.get("name", ""),
        "split": dataset.get("split", ""),
        "samples": accuracy.get("sample_count", 0),
        "size_mb": round(size_bytes / 1_000_000, 4),
        "top1": _round_metric(accuracy.get("top1", "")),
        "top5": _round_metric(accuracy.get("top5", "")),
        "top5_agreement": _round_metric(accuracy.get("top5_agreement", "")),
        "best_median_ms": _round_metric(best_median),
        "best_compute_unit": best_unit,
        "rss_delta_mb": _rss_delta_mb(before_load_rss, after_benchmark_rss),
        "rss_after_benchmark_mb": _bytes_to_mb(after_benchmark_rss),
        "failure_summary": summary.get("failure_summary", ""),
        "run_id": summary.get("run_id", ""),
    }


def regenerate_scoreboard(runs_root: Path, reports_dir: Path) -> list[dict[str, Any]]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_run_summaries(runs_root)
    rows = [flatten_summary(summary) for summary in summaries]
    write_csv(rows, reports_dir / "scoreboard.csv")
    write_markdown(rows, reports_dir / "scoreboard.md")
    real_rows = latest_real_benchmark_rows(summaries)
    write_csv_with_columns(
        real_rows,
        reports_dir / "real_benchmark_table.csv",
        REAL_BENCHMARK_COLUMNS,
    )
    write_markdown_with_columns(
        real_rows,
        reports_dir / "real_benchmark_table.md",
        REAL_BENCHMARK_COLUMNS,
    )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    write_csv_with_columns(rows, path, SCOREBOARD_COLUMNS)


def write_csv_with_columns(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    write_markdown_with_columns(rows, path, SCOREBOARD_COLUMNS)


def write_markdown_with_columns(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latest_real_benchmark_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for summary in summaries:
        if summary.get("execution_mode") != "real":
            continue
        dataset = summary.get("dataset", {})
        if dataset.get("name") != "imagenette":
            continue
        recipe = summary.get("recipe", {})
        key = (
            str(summary.get("model_id", "")),
            str(recipe.get("name", "")),
            str(dataset.get("name", "")),
            str(dataset.get("split", "")),
        )
        previous = latest.get(key)
        if previous is None or summary.get("created_at_unix", 0) >= previous.get(
            "created_at_unix", 0
        ):
            latest[key] = summary

    rows = [flatten_summary(summary) for summary in latest.values()]
    rows.sort(
        key=lambda row: (
            _model_order(row["model_id"]),
            _split_order(row["split"]),
            _recipe_order(row["recipe"]),
            row["model_id"],
            row["recipe"],
        )
    )
    _add_tradeoffs(rows)
    return [{column: row.get(column, "") for column in REAL_BENCHMARK_COLUMNS} for row in rows]


def _round_metric(value: Any) -> Any:
    if value == "":
        return value
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return value


def _bytes_to_mb(value: Any) -> Any:
    if value in ("", None):
        return ""
    try:
        return round(float(value) / 1_000_000, 4)
    except (TypeError, ValueError):
        return ""


def _rss_delta_mb(before: Any, after: Any) -> Any:
    if before in ("", None) or after in ("", None):
        return ""
    try:
        return round((float(after) - float(before)) / 1_000_000, 4)
    except (TypeError, ValueError):
        return ""


def _model_order(model_id: str) -> int:
    order = {
        "apple/mobilevit-xx-small": 0,
        "apple/mobilevit-small": 1,
    }
    return order.get(model_id, 99)


def _split_order(split: str) -> int:
    order = {
        "eval20": 0,
        "eval500": 1,
    }
    return order.get(split, 99)


def _recipe_order(recipe: str) -> int:
    order = {
        "fp16": 0,
        "int8-weight-only": 1,
        "palettized-8bit": 2,
        "palettized-6bit": 3,
        "pruned-sparse-50": 4,
    }
    return order.get(recipe, 99)


def _add_tradeoffs(rows: list[dict[str, Any]]) -> None:
    baselines = {
        (row["model_id"], row["split"]): row
        for row in rows
        if row.get("status") == "ok" and row.get("recipe") == "fp16"
    }
    for row in rows:
        baseline = baselines.get((row["model_id"], row["split"]))
        row["tradeoff"] = _tradeoff(row, baseline)


def _tradeoff(row: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    if row.get("status") != "ok":
        return str(row.get("failure_summary") or row.get("failure_stage") or "run failed")
    if row.get("recipe") == "fp16":
        return "baseline"
    top1 = _as_float(row.get("top1"))
    top5 = _as_float(row.get("top5"))
    if top1 == 0 and top5 == 0:
        return "accuracy collapse"
    if baseline is None:
        return "ok"

    size = _as_float(row.get("size_mb"))
    baseline_size = _as_float(baseline.get("size_mb"))
    median = _as_float(row.get("best_median_ms"))
    baseline_median = _as_float(baseline.get("best_median_ms"))
    baseline_top1 = _as_float(baseline.get("top1"))
    if None in (size, baseline_size, top1, baseline_top1, median, baseline_median):
        return "ok"

    size_delta_pct = ((size / baseline_size) - 1) * 100
    top1_delta = top1 - baseline_top1
    median_delta = median - baseline_median
    return f"size {size_delta_pct:+.1f}%, top1 {top1_delta:+.3f}, median {median_delta:+.3f} ms"


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
