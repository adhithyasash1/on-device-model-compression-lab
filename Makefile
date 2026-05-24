.PHONY: smoke test lint format report

smoke:
	uv run mlclab run configs/smoke/synthetic_fp16.yaml

test:
	uv run pytest

lint:
	uv run ruff format --check .
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

report:
	uv run mlclab report

