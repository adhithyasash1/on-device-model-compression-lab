from pathlib import Path

import pytest

from mlclab.config import CompressionKind, load_config


def test_load_smoke_config() -> None:
    config = load_config(Path("configs/smoke/synthetic_fp16.yaml"))

    assert config.schema_version == 1
    assert config.execution.mode == "synthetic"
    assert config.compression.kind == CompressionKind.FP16
    assert config.compression.calibration_samples is None
    assert config.recovery.logit_repair == "none"
    assert config.benchmark.iterations == 10


def test_mobilevit_benchmark_matrix_configs_load() -> None:
    config_paths = sorted(Path("configs/eval20").glob("*.yaml")) + sorted(
        Path("configs/eval500").glob("*.yaml")
    )

    configs = [load_config(path) for path in config_paths]
    matrix = {
        (config.model.id, config.dataset.split, config.compression.kind) for config in configs
    }

    assert len(configs) == 20
    eval20_limits = {config.dataset.limit for config in configs if config.dataset.split == "eval20"}
    assert eval20_limits == {20}
    eval500_limits = {
        config.dataset.limit for config in configs if config.dataset.split == "eval500"
    }
    assert eval500_limits == {500}
    for model_id in {"apple/mobilevit-xx-small", "apple/mobilevit-small"}:
        for split in {"eval20", "eval500"}:
            assert (model_id, split, CompressionKind.FP16) in matrix
            assert (model_id, split, CompressionKind.INT8_WEIGHT_ONLY) in matrix
            assert (model_id, split, CompressionKind.PALETTIZED) in matrix
            assert (model_id, split, CompressionKind.PRUNED_SPARSE) in matrix


def test_palettized_recipe_requires_bits(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
schema_version: 1
name: bad
model:
  id: synthetic/model
  input_size: 256
dataset:
  name: synthetic
  split: eval20
export: {}
compression:
  name: bad-pal
  kind: palettized
benchmark:
  compute_units: [CPU_ONLY]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="palettized recipes require bits"):
        load_config(config_path)


def test_benchmark_warmup_must_be_positive(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
schema_version: 1
name: bad
model:
  id: synthetic/model
  input_size: 256
dataset:
  name: synthetic
  split: eval20
export: {}
compression:
  name: fp16
  kind: fp16
benchmark:
  warmup: 0
  compute_units: [CPU_ONLY]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="greater than 0"):
        load_config(config_path)


def test_batch_size_above_one_is_rejected_until_supported(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_batch.yaml"
    config_path.write_text(
        """
schema_version: 1
name: bad-batch
execution:
  mode: synthetic
model:
  id: synthetic/model
  input_size: 256
dataset:
  name: synthetic
  split: eval20
export: {}
compression:
  name: fp16
  kind: fp16
benchmark:
  batch_size: 4
  compute_units: [CPU_ONLY]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch_size must be 1"):
        load_config(config_path)


def test_fp16_recipe_rejects_activation_calibration(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_calibration.yaml"
    config_path.write_text(
        """
schema_version: 1
name: bad-calibration
execution:
  mode: real
model:
  id: apple/mobilevit-xx-small
  input_size: 256
dataset:
  name: imagenette
  split: eval20
  root: data/imagenette2-160/val
export: {}
compression:
  name: fp16
  kind: fp16
  calibration_samples: 32
benchmark:
  compute_units: [CPU_ONLY]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="calibration_samples require"):
        load_config(config_path)


def test_logit_repair_requires_calibration_samples(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_repair.yaml"
    config_path.write_text(
        """
schema_version: 1
name: bad-repair
execution:
  mode: real
model:
  id: apple/mobilevit-xx-small
  input_size: 256
dataset:
  name: imagenette
  split: eval20
  root: data/imagenette2-160/val
export: {}
compression:
  name: int8-weight-only
  kind: int8_weight_only
  bits: 8
recovery:
  logit_repair: class_bias
benchmark:
  compute_units: [CPU_ONLY]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="logit_repair requires calibration_samples"):
        load_config(config_path)


def test_recovery_sweep_configs_load() -> None:
    config_paths = sorted(Path("configs/recovery").glob("*.yaml"))
    configs = [load_config(path) for path in config_paths]

    assert {config.compression.calibration_samples for config in configs} == {32, 128, 512}
    assert all(config.recovery.logit_repair == "class_bias" for config in configs)
