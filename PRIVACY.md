# Privacy

This project is a local CLI. It does not collect telemetry and does not upload datasets, model weights, logs, machine identifiers, or benchmark results.

Commands may download models or datasets only when the user runs a path that explicitly needs them. Generated run folders stay local unless the user shares them.

Committed reports are generated with stable run IDs instead of absolute local paths. Ignored run folders may still contain local paths in `summary.json`, `environment.json`, and debug logs because those files are meant to stay on the machine that produced the benchmark.
