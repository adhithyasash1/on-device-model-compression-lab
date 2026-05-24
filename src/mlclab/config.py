from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComputeUnit(StrEnum):
    CPU_ONLY = "CPU_ONLY"
    CPU_AND_GPU = "CPU_AND_GPU"
    CPU_AND_NE = "CPU_AND_NE"
    ALL = "ALL"


class CompressionKind(StrEnum):
    FP16 = "fp16"
    INT8_WEIGHT_ONLY = "int8_weight_only"
    PALETTIZED = "palettized"
    PRUNED_SPARSE = "pruned_sparse"


class ExecutionSpec(StrictModel):
    mode: Literal["synthetic", "real"] = "real"
    strict_scoreboard: bool = True


class ModelSpec(StrictModel):
    id: str
    revision: str | None = None
    input_size: int = Field(gt=0)
    num_labels: int = Field(default=1000, gt=0)


class DatasetSpec(StrictModel):
    name: Literal["imagenette", "imagenet-local", "synthetic"]
    split: str
    seed: int = 1337
    root: Path | None = None
    manifest_path: Path | None = None
    limit: int | None = Field(default=None, gt=0)


class ExportSpec(StrictModel):
    format: Literal["coreml"] = "coreml"
    input_type: Literal["tensor", "image"] = "tensor"
    deployment_target: str = "macOS15"
    precision: Literal["fp16", "fp32"] = "fp16"
    convert_to: Literal["mlprogram"] = "mlprogram"


class CompressionRecipe(StrictModel):
    name: str
    kind: CompressionKind
    bits: int | None = None
    mode: str | None = None
    target_sparsity: float | None = None
    calibration_samples: int | None = Field(default=None, gt=0)
    calibration_seed: int | None = None
    calibration_op_group_size: int = -1

    @field_validator("bits")
    @classmethod
    def validate_bits(cls, value: int | None) -> int | None:
        if value is not None and value not in {1, 2, 3, 4, 6, 8}:
            raise ValueError("bits must be one of 1, 2, 3, 4, 6, or 8")
        return value

    @field_validator("target_sparsity")
    @classmethod
    def validate_sparsity(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 1:
            raise ValueError("target_sparsity must be between 0 and 1")
        return value

    @field_validator("calibration_op_group_size")
    @classmethod
    def validate_calibration_op_group_size(cls, value: int) -> int:
        if value == 0 or value < -1:
            raise ValueError("calibration_op_group_size must be -1 or a positive integer")
        return value

    @model_validator(mode="after")
    def validate_kind_specific_fields(self) -> CompressionRecipe:
        if self.kind == CompressionKind.PALETTIZED and self.bits is None:
            raise ValueError("palettized recipes require bits")
        if self.kind == CompressionKind.PRUNED_SPARSE and self.target_sparsity is None:
            raise ValueError("pruned_sparse recipes require target_sparsity")
        if self.calibration_samples is not None and self.kind == CompressionKind.FP16:
            raise ValueError("calibration_samples require a compressed Core ML recipe")
        return self


class RecoverySpec(StrictModel):
    logit_repair: Literal["none", "class_bias"] = "none"
    calibration_samples: int | None = Field(default=None, gt=0)
    calibration_seed: int | None = None

    @model_validator(mode="after")
    def validate_recovery(self) -> RecoverySpec:
        if self.logit_repair != "none" and self.calibration_samples is None:
            raise ValueError("logit_repair requires calibration_samples")
        return self


class BenchmarkSpec(StrictModel):
    batch_size: int = Field(default=1, gt=0)
    warmup: int = Field(default=10, gt=0)
    iterations: int = Field(default=100, gt=0)
    compute_units: list[ComputeUnit]

    @field_validator("compute_units")
    @classmethod
    def validate_compute_units(cls, value: list[ComputeUnit]) -> list[ComputeUnit]:
        if not value:
            raise ValueError("at least one compute unit is required")
        return value

    @model_validator(mode="after")
    def validate_supported_batch_size(self) -> BenchmarkSpec:
        if self.batch_size != 1:
            raise ValueError("batch_size must be 1 until batched benchmarking is implemented")
        return self


class RunConfig(StrictModel):
    schema_version: Literal[1]
    name: str
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    model: ModelSpec
    dataset: DatasetSpec
    export: ExportSpec
    compression: CompressionRecipe
    recovery: RecoverySpec = Field(default_factory=RecoverySpec)
    benchmark: BenchmarkSpec
    output_dir: Path = Path("runs")
    reports_dir: Path = Path("reports")

    def to_plain_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_config(path: Path) -> RunConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return RunConfig.model_validate(raw)


def dump_config(config: RunConfig, path: Path) -> None:
    path.write_text(yaml.safe_dump(config.to_plain_dict(), sort_keys=False), encoding="utf-8")
