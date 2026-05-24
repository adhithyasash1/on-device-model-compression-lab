from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlclab.artifacts.environment import write_environment
from mlclab.config import RunConfig, dump_config


@dataclass(frozen=True)
class RunContext:
    root: Path
    artifacts_dir: Path
    logs_dir: Path
    metrics_path: Path
    summary_path: Path


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


class RunStore:
    def __init__(self, runs_root: Path) -> None:
        self.runs_root = runs_root

    def create(self, config: RunConfig, repo_root: Path) -> RunContext:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        model = _slug(config.model.id.split("/")[-1])
        recipe = _slug(config.compression.name)
        base_name = f"{timestamp}_{model}_{recipe}"
        run_root = self.runs_root / base_name
        suffix = 1
        while run_root.exists():
            suffix += 1
            run_root = self.runs_root / f"{base_name}_{suffix}"

        artifacts_dir = run_root / "artifacts"
        logs_dir = run_root / "logs"
        artifacts_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)

        context = RunContext(
            root=run_root,
            artifacts_dir=artifacts_dir,
            logs_dir=logs_dir,
            metrics_path=run_root / "metrics.jsonl",
            summary_path=run_root / "summary.json",
        )
        dump_config(config, run_root / "config.yaml")
        write_environment(run_root / "environment.json", repo_root)
        self.log_event(context, "run_created", {"run_dir": str(run_root)})
        return context

    def log_event(
        self, context: RunContext, event: str, payload: dict[str, Any] | None = None
    ) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": payload or {},
        }
        with (context.logs_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def append_metric(self, context: RunContext, record: dict[str, Any]) -> None:
        with context.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def write_summary(self, context: RunContext, summary: dict[str, Any]) -> None:
        context.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
