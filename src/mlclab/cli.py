from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from mlclab.config import load_config
from mlclab.data import DatasetPreparationError, prepare_imagenette160
from mlclab.debug import debug_sample
from mlclab.pipeline import plan_run, run_config
from mlclab.reports import regenerate_scoreboard

app = typer.Typer(help="MobileViT Core ML compression lab.")
debug_app = typer.Typer(help="Correctness and artifact debug tools.")
app.add_typer(debug_app, name="debug")
console = Console()


@app.command()
def validate(config_path: Path) -> None:
    """Validate a run config."""
    config = load_config(config_path)
    console.print(f"ok: {config.name}")


@app.command()
def plan(config_path: Path, repo_root: Path = Path(".")) -> None:
    """Print the resolved execution plan without creating artifacts."""
    config = load_config(config_path)
    console.print_json(json.dumps(plan_run(config, repo_root.resolve())))


@app.command()
def run(config_path: Path, repo_root: Path = Path("."), dry_run: bool = False) -> None:
    """Run one benchmark config."""
    config = load_config(config_path)
    if dry_run:
        console.print_json(json.dumps(plan_run(config, repo_root.resolve())))
        return

    summary = run_config(config, repo_root.resolve())
    console.print(f"status: {summary['status']}")
    console.print(f"run_dir: {summary['run_dir']}")
    console.print(f"scoreboard: {repo_root.resolve() / config.reports_dir / 'scoreboard.md'}")
    if config.execution.strict_scoreboard and summary["status"] != "ok":
        raise typer.Exit(1)


@app.command("prepare-imagenette160")
def prepare_imagenette160_command(
    root: Annotated[
        Path,
        typer.Option(
            "--root",
            help="Imagenette 160px dataset root, or an existing validation folder.",
        ),
    ] = Path("data/imagenette2-160"),
    manifests_dir: Annotated[
        Path,
        typer.Option(
            "--manifests-dir",
            help="Directory for eval20 and eval500 manifest files.",
        ),
    ] = Path("data/manifests"),
    download: Annotated[
        bool,
        typer.Option(
            "--download/--no-download",
            help="Download Imagenette 160px when the dataset root is missing.",
        ),
    ] = True,
) -> None:
    """Download or validate Imagenette 160px and write fixed eval manifests."""
    try:
        result = prepare_imagenette160(root, manifests_dir=manifests_dir, download=download)
    except (DatasetPreparationError, FileNotFoundError) as exc:
        console.print(f"error: {exc}")
        raise typer.Exit(1) from exc

    console.print(f"dataset_root: {result['dataset_root']}")
    console.print(f"val_root: {result['val_root']}")
    for manifest in result["manifests"]:
        console.print(
            f"{manifest['split']}: {manifest['path']} "
            f"samples={manifest['sample_count']} sha256={manifest['sha256']}"
        )
        console.print(f"{manifest['split']}_metadata: {manifest['metadata_path']}")


@app.command()
def report(runs_root: Path = Path("runs"), reports_dir: Path = Path("reports")) -> None:
    """Regenerate scoreboard files from run summaries."""
    rows = regenerate_scoreboard(runs_root, reports_dir)
    console.print(f"wrote {len(rows)} rows to {reports_dir / 'scoreboard.md'}")


@debug_app.command("sample")
def debug_sample_command(
    run_dir: Annotated[
        Path,
        typer.Option("--run", help="Run directory containing config.yaml and summary.json."),
    ],
    sample_id: Annotated[
        str,
        typer.Option("--sample-id", help="Dataset sample_id from the run manifest."),
    ],
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repo root for resolving relative dataset paths."),
    ] = None,
    compute_unit: Annotated[
        str | None,
        typer.Option("--compute-unit", help="Core ML compute unit, defaults to the run config."),
    ] = None,
    dump_logits: Annotated[
        bool,
        typer.Option("--dump-logits/--no-dump-logits", help="Write full logits JSON."),
    ] = False,
    logits_path: Annotated[
        Path | None,
        typer.Option("--logits-path", help="Path for dumped logits JSON."),
    ] = None,
) -> None:
    """Inspect one sample through PyTorch and Core ML."""
    try:
        result = debug_sample(
            run_dir,
            sample_id,
            repo_root=repo_root,
            compute_unit=compute_unit,
            dump_logits=dump_logits,
            logits_path=logits_path,
        )
    except Exception as exc:
        console.print(f"error: {exc}")
        raise typer.Exit(1) from exc

    console.print(f"run_dir: {result['run_dir']}")
    console.print(f"artifact: {result['artifact_path']}")
    console.print(f"compute_unit: {result['compute_unit']}")
    label = f"{result['label']}"
    if result.get("synset"):
        label = f"{label} ({result['synset']})"
    console.print(f"sample_id: {result['sample_id']}")
    console.print(f"label: {label}")
    stats = result["tensor_stats"]
    console.print(
        "tensor: "
        f"shape={stats['shape']} dtype={stats['dtype']} "
        f"min={stats['min']:.6f} max={stats['max']:.6f} "
        f"mean={stats['mean']:.6f} std={stats['std']:.6f}"
    )
    _print_top5("PyTorch top-5", result["pytorch_top5"])
    _print_top5("Core ML top-5", result["coreml_top5"])
    if "coreml_repaired_top5" in result:
        _print_top5("Core ML repaired top-5", result["coreml_repaired_top5"])
    if "logits_path" in result:
        console.print(f"logits: {result['logits_path']}")


def _print_top5(title: str, rows: list[dict[str, float | int]]) -> None:
    console.print(f"{title}:")
    for row in rows:
        console.print(f"  {row['rank']}. class={row['class_id']} logit={float(row['logit']):.6f}")


if __name__ == "__main__":
    app()
