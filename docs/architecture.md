# Architecture

The CLI is intentionally thin. It loads a config, validates it, then calls the pipeline layer.

```text
cli.py
  -> config.py
  -> pipeline/run.py
       -> data/
       -> models/
       -> export/
       -> compress/
       -> benchmark/
       -> metrics/
       -> artifacts/
       -> reports/
```

The stable v1 public API is the CLI plus the config schema. Python internals can move while the lab is forming.

## Run Contract

Each run writes an immutable folder:

```text
runs/<timestamp>_<model>_<recipe>/
  config.yaml
  environment.json
  metrics.jsonl
  summary.json
  artifacts/
  logs/
```

Reports are regenerated from run summaries.

## Report Contract

`mlclab report` scans `runs/*/summary.json`, writes the ignored full scoreboard, and writes the committed real benchmark table:

```text
reports/scoreboard.csv
reports/scoreboard.md
reports/real_benchmark_table.csv
reports/real_benchmark_table.md
```

The full scoreboard is ignored because it can contain exploratory failures. The real benchmark table keeps the latest real Imagenette row for each `(model, recipe, split)` combination and uses `run_id` instead of absolute local run paths.

Recovery runs can also write:

```text
runs/<timestamp>_<model>_<recipe>/artifacts/class_bias_repair.json
```

That file stores the fitted post-quantization class-bias vector used to report
repaired logits. The raw Core ML top-5 is still recorded in prediction metrics
when repair is enabled.
