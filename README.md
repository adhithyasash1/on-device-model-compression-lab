# On-Device Model Compression Lab

Thesis: compression decisions for on-device vision models should be made from measured size, accuracy, latency, and memory tradeoffs, not from recipe names. This repo benchmarks MobileViT Core ML compression lanes locally on Mac and keeps the measured artifacts reproducible.

The current decision split is Imagenette `eval500`. The best measured default is `apple/mobilevit-small` with `int8-weight-only`: 49.0% smaller than FP16, top-1 moved from 0.728 to 0.740, top-5 stayed at 0.954, and median latency moved from 1.8174 ms to 1.8100 ms on the fastest requested Core ML compute unit.

## Benchmark Snapshot

Real Core ML rows from [`reports/real_benchmark_table.csv`](reports/real_benchmark_table.csv), filtered to `eval500`.

| Model | Recipe | Size MB | Size drop | Top-1 | Top-5 | Median ms | RSS after MB | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MobileViT xx-small | `fp16` | 2.8852 | baseline | 0.600 | 0.864 | 1.0771 | 870.2525 | Baseline |
| MobileViT xx-small | `int8-weight-only` | 1.5172 | 47.4% | 0.592 | 0.858 | 1.1158 | 965.5419 | Good size, slower |
| MobileViT xx-small | `palettized-8bit` | 1.5387 | 46.7% | 0.598 | 0.860 | 1.0770 | 1034.5841 | Best xx-small balance |
| MobileViT xx-small | `palettized-6bit` | 1.1991 | 58.4% | 0.374 | 0.618 | 1.0821 | 844.4969 | Size stress test |
| MobileViT xx-small | `pruned-sparse-50` | 1.6568 | 42.6% | 0.000 | 0.000 | 1.0730 | 998.5393 | Failed accuracy |
| MobileViT small | `fp16` | 11.4904 | baseline | 0.728 | 0.954 | 1.8174 | 1078.9519 | Baseline |
| MobileViT small | `int8-weight-only` | 5.8558 | 49.0% | 0.740 | 0.954 | 1.8100 | 1112.2115 | Best measured default |
| MobileViT small | `palettized-8bit` | 5.8594 | 49.0% | 0.726 | 0.946 | 1.8471 | 701.6120 | Accurate, slower |
| MobileViT small | `palettized-6bit` | 4.4418 | 61.3% | 0.682 | 0.918 | 2.0203 | 946.7003 | Size-first only |
| MobileViT small | `pruned-sparse-50` | 6.5077 | 43.4% | 0.000 | 0.000 | 1.8891 | 1033.3716 | Failed accuracy |

`RSS after MB` is process RSS after benchmarking, not isolated model peak memory.

## Key Findings

- `mobilevit-small` plus `int8-weight-only` is the strongest measured shipping candidate on `eval500`: 49.0% size reduction, +0.012 top-1, unchanged top-5, and -0.007 ms median latency versus FP16.
- `palettized-6bit` gives the smallest artifacts, 58.4% smaller for xx-small and 61.3% smaller for small, but its accuracy loss is material.
- `palettized-8bit` preserves accuracy closely, but it does not beat `int8-weight-only` on the small model.
- `pruned-sparse-50` collapses both models to 0.000 top-1 and 0.000 top-5 without fine-tuning or recovery.
- No recipe improved latency reliably across both model sizes.

## Methodology

The real path loads MobileViT weights through Transformers, exports a tensor-input FP16 Core ML `mlprogram`, applies Core ML compression recipes, then evaluates Core ML predictions against ImageNet-1k labels mapped from Imagenette synsets. Latency measures only prepared Core ML `predict` calls after warmup. The benchmark records top-1, top-5, top-5 agreement with PyTorch, artifact size, median latency, and process RSS.

The benchmark configs request `CPU_ONLY`, `CPU_AND_GPU`, and `ALL`; the table reports the fastest requested unit per row. `ALL` is a Core ML compute-unit request, not proof of ANE execution. There is no iPhone runner in this version.

## How To Reproduce

Install development dependencies:

```bash
uv sync --group dev
```

Run checks and the machinery-only smoke path:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run mlclab run configs/smoke/synthetic_fp16.yaml
uv run mlclab report
```

Prepare the real MobileViT path:

```bash
uv sync --extra ml --group dev
uv run mlclab prepare-imagenette160
```

Run the `eval500` benchmark matrix:

```bash
for config in configs/eval500/*.yaml; do
  uv run mlclab run "$config"
done
uv run mlclab report
```

Inspect one real run:

```bash
uv run mlclab debug sample --run runs/<run_id> --sample-id <sample_id>
uv run mlclab debug sample --run runs/<run_id> --sample-id <sample_id> --dump-logits
```

## Artifacts And Docs

- Case study: [`reports/case_study.md`](reports/case_study.md)
- Full real benchmark table: [`reports/real_benchmark_table.md`](reports/real_benchmark_table.md)
- Environment note: [`reports/benchmark_environment.md`](reports/benchmark_environment.md)
- Charts: [`reports/figures/size_vs_accuracy.png`](reports/figures/size_vs_accuracy.png), [`reports/figures/latency_vs_accuracy.png`](reports/figures/latency_vs_accuracy.png)
- Architecture: [`docs/architecture.md`](docs/architecture.md)
- Dataset notes: [`DATASETS.md`](DATASETS.md)
- Model license notes: [`MODEL_LICENSES.md`](MODEL_LICENSES.md)
- Privacy notes: [`PRIVACY.md`](PRIVACY.md)
- Configs: [`configs/eval500/`](configs/eval500/), [`configs/eval20/`](configs/eval20/), [`configs/recovery/`](configs/recovery/)
- Recipes: [`recipes/`](recipes/)
- Key modules: [`src/mlclab/pipeline/run.py`](src/mlclab/pipeline/run.py), [`src/mlclab/reports/scoreboard.py`](src/mlclab/reports/scoreboard.py), [`src/mlclab/compress/coreml.py`](src/mlclab/compress/coreml.py), [`src/mlclab/benchmark/coreml_runner.py`](src/mlclab/benchmark/coreml_runner.py)

## Project Structure

```text
configs/     Benchmark and recovery configs.
recipes/     Compression recipe templates.
src/mlclab/  CLI, config schema, data prep, export, compression, benchmarking, reporting.
tests/        Unit and smoke tests for config loading, metrics, reporting, CLI, and recovery.
reports/      Committed benchmark tables, case study, charts, and environment note.
runs/         Ignored local run folders with summaries, metrics, logs, and artifacts.
data/         Ignored local datasets and manifests.
```

## Caveats

- Accuracy is measured on Imagenette subsets, not full ImageNet-1k validation.
- The decision split has 500 images. It is useful for local iteration, not a final product acceptance test.
- Results come from one local arm64 macOS host. Use [`reports/benchmark_environment.md`](reports/benchmark_environment.md) to interpret them.
- The compressed `.mlpackage` artifacts and downloaded model weights are ignored and must be regenerated locally.
- Apple MobileViT weights have separate upstream terms. See [`MODEL_LICENSES.md`](MODEL_LICENSES.md).

## Known Limitations

- Batch size is fixed at 1.
- The benchmark reports process RSS after the run, not model-only peak resident memory.
- The pruning lane has no fine-tuning, so the current `pruned-sparse-50` result is a failure row.
- Core ML compile and model loading are outside timed latency samples.
- The recovery configs are experimental and are not part of the decision table yet.

## Next Steps

- Add a full ImageNet-1k validation mode with explicit dataset-root configuration.
- Add repeated-run aggregation with confidence intervals for latency and accuracy deltas.
- Add an iPhone benchmark harness before making device claims.
- Add fine-tuning or structured recovery for sparse pruning.
- Add report validation that fails if committed benchmark tables diverge from run summaries.

## Sources

- [`reports/real_benchmark_table.csv`](reports/real_benchmark_table.csv)
- [`reports/case_study.md`](reports/case_study.md)
- [`reports/benchmark_environment.md`](reports/benchmark_environment.md)
- [`configs/eval500/`](configs/eval500/)
- [`src/mlclab/pipeline/run.py`](src/mlclab/pipeline/run.py)
- [`src/mlclab/reports/scoreboard.py`](src/mlclab/reports/scoreboard.py)
- [`src/mlclab/data/imagenette.py`](src/mlclab/data/imagenette.py)
