"""Typed, portable experiment configuration for PTU-Net.

Paths in a configuration file are resolved relative to that file, not to the
caller's working directory.  This makes the same YAML usable from the CLI,
tests, schedulers, and notebooks.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

DEFAULT_COMPRESSOR_EXTENSIONS: dict[str, str] = {
    "sz3": ".sz3.out",
    "sz": ".sz.out",
    "szp": ".szp.out",
    "sperr": ".sperr.out",
    "hpez": ".hpez.out",
    "zfp": ".zfp.out",
    "mgard": ".mgard.out",
    "fpzip": ".fpzip.out",
}


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")


def _non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer, got {value!r}")


def _positive_float(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigError(f"{name} must be positive, got {value!r}")


def _non_negative_float(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ConfigError(f"{name} must be non-negative, got {value!r}")


@dataclass(frozen=True)
class PathConfig:
    """Machine-local roots, resolved by :func:`load_config`."""

    data_root: Path
    output_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "output_root", Path(self.output_root))


@dataclass(frozen=True)
class DatasetSpec:
    """File layout and temporal splits for one scientific field."""

    name: str
    variable: str
    resolution_tag: str
    height: int
    width: int
    train_timesteps: tuple[int, ...]
    validation_timesteps: tuple[int, ...] = ()
    test_timesteps: tuple[int, ...] = ()
    even_compressor: str = "sz3"
    odd_compressor: str = "sz3"
    timestep_digits: int = 4
    output_subdir: str = "{name}/{compressor}"
    original_template: str = "{variable}_{resolution_tag}_{timestep}.dat"
    compressed_template: str = "{variable}_{resolution_tag}_{timestep}.dat{extension}"
    reconstruction_template: str = "{variable}_{resolution_tag}_{timestep}.dat{extension}.recon.dat"

    def __post_init__(self) -> None:
        for field_name in ("name", "variable", "resolution_tag"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"dataset {field_name} must be a non-empty string")
        _positive_int("dataset height", self.height)
        _positive_int("dataset width", self.width)
        _positive_int("timestep_digits", self.timestep_digits)

        for field_name in (
            "train_timesteps",
            "validation_timesteps",
            "test_timesteps",
        ):
            values = tuple(getattr(self, field_name))
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise ConfigError(f"{field_name} must contain only integers")
            if any(value < 0 for value in values):
                raise ConfigError(f"{field_name} cannot contain negative timesteps")
            if len(set(values)) != len(values):
                raise ConfigError(f"{field_name} contains duplicate timesteps")
            object.__setattr__(self, field_name, values)

        if not self.train_timesteps:
            raise ConfigError(f"dataset {self.name!r} must have at least one training timestep")
        split_sets = {
            "train": set(self.train_timesteps),
            "validation": set(self.validation_timesteps),
            "test": set(self.test_timesteps),
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ConfigError(
                    f"dataset {self.name!r} has overlapping {left}/{right} timesteps: "
                    f"{sorted(overlap)}"
                )

        for field_name in (
            "even_compressor",
            "odd_compressor",
            "output_subdir",
            "original_template",
            "compressed_template",
            "reconstruction_template",
        ):
            if not isinstance(getattr(self, field_name), str):
                raise ConfigError(f"{field_name} must be a string")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def all_timesteps(self) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(self.train_timesteps + self.validation_timesteps + self.test_timesteps)
        )

    def timesteps_for_split(self, split: str) -> tuple[int, ...]:
        normalized = split.lower()
        if normalized == "train":
            return self.train_timesteps
        if normalized in {"validation", "val"}:
            return self.validation_timesteps
        if normalized == "test":
            return self.test_timesteps
        raise KeyError(f"unknown split {split!r}; expected train, validation, or test")

    def compressor_for_timestep(self, timestep: int) -> str:
        return self.even_compressor if timestep % 2 == 0 else self.odd_compressor


@dataclass(frozen=True)
class ModelConfig:
    """Model constructor fields from ``PTUNetConfig``, plus data patch stride."""

    patch_size: int = 32
    stride: int = 16
    subpatch_size: int = 8
    embed_dim: int = 384
    num_heads: int = 12
    num_layers: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    input_mode: str = "center_difference"

    baseline_prior: tuple[float, float, float] = (0.05, 0.90, 0.05)
    baseline_hidden_channels: int = 8
    baseline_mix_init: float = 0.5
    use_adaptive_baseline: bool = True
    learnable_baseline_prior: bool = True

    use_transformer_correction: bool = True
    use_positional_embedding: bool = True
    correction_scale_init: float = 0.02
    correction_scale_max: float = 0.2

    use_unet_refinement: bool = True
    unet_base_channels: int = 24
    unet_scale_init: float = 0.4
    use_skip_attention: bool = True
    use_squeeze_excite: bool = True
    squeeze_excite_reduction: int = 4

    linear_init_gain: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "patch_size",
            "stride",
            "subpatch_size",
            "embed_dim",
            "num_heads",
            "baseline_hidden_channels",
            "unet_base_channels",
            "squeeze_excite_reduction",
        ):
            _positive_int(f"model.{name}", getattr(self, name))
        _non_negative_int("model.num_layers", self.num_layers)
        if self.stride > self.patch_size:
            raise ConfigError("model.stride cannot exceed patch_size (it would leave gaps)")
        if self.patch_size % self.subpatch_size:
            raise ConfigError("model.patch_size must be divisible by subpatch_size")
        if self.embed_dim % self.num_heads:
            raise ConfigError("model.embed_dim must be divisible by num_heads")
        _positive_float("model.mlp_ratio", self.mlp_ratio)
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0 <= self.dropout < 1
        ):
            raise ConfigError("model.dropout must be in [0, 1)")
        if self.input_mode not in {"center_difference", "raw_frames"}:
            raise ConfigError("model.input_mode must be center_difference or raw_frames")
        prior = tuple(self.baseline_prior)
        if len(prior) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in prior
        ):
            raise ConfigError("model.baseline_prior must contain three positive values")
        object.__setattr__(self, "baseline_prior", prior)
        if (
            isinstance(self.baseline_mix_init, bool)
            or not isinstance(self.baseline_mix_init, (int, float))
            or not 0 < self.baseline_mix_init < 1
        ):
            raise ConfigError("model.baseline_mix_init must be in (0, 1)")
        _positive_float("model.correction_scale_max", self.correction_scale_max)
        _positive_float("model.correction_scale_init", self.correction_scale_init)
        if not 0 < self.correction_scale_init < self.correction_scale_max:
            raise ConfigError(
                "model.correction_scale_init must be between zero and correction_scale_max"
            )
        _positive_float("model.linear_init_gain", self.linear_init_gain)
        for name in (
            "use_adaptive_baseline",
            "learnable_baseline_prior",
            "use_transformer_correction",
            "use_positional_embedding",
            "use_unet_refinement",
            "use_skip_attention",
            "use_squeeze_excite",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"model.{name} must be a boolean")


@dataclass(frozen=True)
class LossConfig:
    """YAML-friendly counterpart of ``ptunet.losses.LossWeights``."""

    mse: float = 0.7
    charbonnier: float = 0.3
    correction: float = 1.0e-4
    gradient: float = 0.0
    spectral: float = 0.0

    def __post_init__(self) -> None:
        values = (self.mse, self.charbonnier, self.correction, self.gradient, self.spectral)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ConfigError("training.loss weights must be non-negative")
        if self.mse + self.charbonnier + self.gradient + self.spectral <= 0:
            raise ConfigError("at least one reconstruction loss weight must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 300
    batch_size: int = 32
    validation_batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-3
    gradient_clip_norm: float = 0.5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    minimum_learning_rate: float = 1.0e-7
    early_stopping_patience: int = 15
    minimum_delta: float = 1.0e-6
    gradient_accumulation_steps: int = 1
    baseline_warmup_epochs: int = 5
    mixed_precision: bool = True
    normalization_chunk_elements: int = 1_048_576
    seed: int = 42
    device: str = "auto"
    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "validation_batch_size"):
            _positive_int(f"training.{name}", getattr(self, name))
        for name in ("num_workers", "scheduler_patience", "baseline_warmup_epochs", "seed"):
            _non_negative_int(f"training.{name}", getattr(self, name))
        _positive_int("training.early_stopping_patience", self.early_stopping_patience)
        _positive_int("training.gradient_accumulation_steps", self.gradient_accumulation_steps)
        _positive_int("training.normalization_chunk_elements", self.normalization_chunk_elements)
        for name in (
            "learning_rate",
            "gradient_clip_norm",
        ):
            _positive_float(f"training.{name}", getattr(self, name))
        _non_negative_float("training.minimum_learning_rate", self.minimum_learning_rate)
        _non_negative_float("training.weight_decay", self.weight_decay)
        if (
            isinstance(self.scheduler_factor, bool)
            or not isinstance(self.scheduler_factor, (int, float))
            or not 0 < self.scheduler_factor < 1
        ):
            raise ConfigError("training.scheduler_factor must be in (0, 1)")
        _non_negative_float("training.minimum_delta", self.minimum_delta)
        if not isinstance(self.device, str) or not self.device:
            raise ConfigError("training.device must be a non-empty string")
        if not isinstance(self.mixed_precision, bool):
            raise ConfigError("training.mixed_precision must be a boolean")
        if isinstance(self.loss, Mapping):
            raw_loss = dict(self.loss)
            _unexpected("training.loss", raw_loss, set(LossConfig.__dataclass_fields__))
            object.__setattr__(self, "loss", LossConfig(**raw_loss))
        elif not isinstance(self.loss, LossConfig):
            raise ConfigError("training.loss must be a mapping")


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 8
    num_workers: int = 4
    temporal_search_radius: int = 2
    metric_chunk_elements: int = 1_048_576
    data_range: float | None = None

    def __post_init__(self) -> None:
        _positive_int("evaluation.batch_size", self.batch_size)
        _non_negative_int("evaluation.num_workers", self.num_workers)
        _positive_int("evaluation.temporal_search_radius", self.temporal_search_radius)
        _positive_int("evaluation.metric_chunk_elements", self.metric_chunk_elements)
        if self.data_range is not None:
            _positive_float("evaluation.data_range", self.data_range)


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    project: str = "ptu-net"
    run_name: str | None = None
    entity: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigError("tracking.enabled must be a boolean")
        if not isinstance(self.project, str) or not self.project:
            raise ConfigError("tracking.project must be a non-empty string")
        for name in ("run_name", "entity"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ConfigError(f"tracking.{name} must be a string or null")
        if isinstance(self.tags, (str, bytes)):
            raise ConfigError("tracking.tags must be a list of strings")
        tags = tuple(self.tags)
        if any(not isinstance(tag, str) or not tag for tag in tags):
            raise ConfigError("tracking.tags must contain non-empty strings")
        object.__setattr__(self, "tags", tags)


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration consumed by data, training, and evaluation code."""

    paths: PathConfig
    datasets: tuple[DatasetSpec, ...]
    name: str = "ptunet-experiment"
    compressor_extensions: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_COMPRESSOR_EXTENSIONS)
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def __post_init__(self) -> None:
        datasets = tuple(self.datasets)
        object.__setattr__(self, "datasets", datasets)
        if not isinstance(self.name, str) or not self.name:
            raise ConfigError("experiment name must be a non-empty string")
        if not datasets:
            raise ConfigError("at least one dataset must be configured")
        names = [dataset.name for dataset in datasets]
        if len(set(names)) != len(names):
            raise ConfigError("dataset names must be unique")

        extensions = dict(self.compressor_extensions)
        if not extensions:
            raise ConfigError("compressor_extensions cannot be empty")
        for compressor, extension in extensions.items():
            if not isinstance(compressor, str) or not compressor:
                raise ConfigError("compressor names must be non-empty strings")
            if not isinstance(extension, str):
                raise ConfigError(f"extension for compressor {compressor!r} must be a string")
        object.__setattr__(self, "compressor_extensions", extensions)
        for dataset in datasets:
            for compressor in (dataset.even_compressor, dataset.odd_compressor):
                if compressor not in extensions:
                    raise ConfigError(
                        f"dataset {dataset.name!r} refers to unknown compressor {compressor!r}"
                    )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, reloadable YAML representation.

        The dataclasses deliberately use convenient Python field names such as
        ``compressor_extensions`` and ``train_timesteps``.  The public YAML
        schema groups those values under ``compressors``, ``splits``, and
        ``templates``.  Serializing that schema here ensures that output from
        ``show-config`` and saved ``config.resolved.yaml`` files can be passed
        back to :func:`load_config` unchanged.

        Paths remain absolute because they have already been resolved relative
        to the source configuration.  A dumped configuration therefore keeps
        pointing to the same data and output roots even when it is reloaded
        from another directory.
        """

        datasets: list[dict[str, Any]] = []
        for dataset in self.datasets:
            datasets.append(
                {
                    "name": dataset.name,
                    "variable": dataset.variable,
                    "resolution_tag": dataset.resolution_tag,
                    "shape": [dataset.height, dataset.width],
                    "timestep_digits": dataset.timestep_digits,
                    "splits": {
                        "train": list(dataset.train_timesteps),
                        "validation": list(dataset.validation_timesteps),
                        "test": list(dataset.test_timesteps),
                    },
                    "compressors": {
                        "even": dataset.even_compressor,
                        "odd": dataset.odd_compressor,
                    },
                    "output_subdir": dataset.output_subdir,
                    "templates": {
                        "original": dataset.original_template,
                        "compressed": dataset.compressed_template,
                        "reconstruction": dataset.reconstruction_template,
                    },
                }
            )

        model = asdict(self.model)
        model["baseline_prior"] = list(self.model.baseline_prior)
        tracking = asdict(self.tracking)
        tracking["tags"] = list(self.tracking.tags)

        return {
            "name": self.name,
            "paths": {
                "data_root": str(self.paths.data_root),
                "output_root": str(self.paths.output_root),
            },
            "compressors": {"extensions": dict(self.compressor_extensions)},
            "datasets": datasets,
            "model": model,
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "tracking": tracking,
        }


def _unexpected(section: str, values: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"unknown key(s) in {section}: {', '.join(unknown)}")


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{section} must be a mapping")
    return dict(value)


def _resolve_path(value: Any, base_dir: Path, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ConfigError(f"paths.{name} must be a path string")
    expanded = os.path.expanduser(os.path.expandvars(os.fspath(value)))
    if "$" in expanded:
        raise ConfigError(f"paths.{name} contains an unresolved environment variable")
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _parse_timesteps(value: Any, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        values = dict(value)
        _unexpected(name, values, {"start", "end", "step"})
        if "start" not in values or "end" not in values:
            raise ConfigError(f"{name} range requires start and end")
        start, end, step = values["start"], values["end"], values.get("step", 1)
        for key, item in (("start", start), ("end", end), ("step", step)):
            if isinstance(item, bool) or not isinstance(item, int):
                raise ConfigError(f"{name}.{key} must be an integer")
        if step <= 0:
            raise ConfigError(f"{name}.step must be positive")
        if end < start:
            raise ConfigError(f"{name}.end must be at least start")
        return tuple(range(start, end + 1, step))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(value)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
            raise ConfigError(f"{name} must contain only integers")
        return result
    raise ConfigError(f"{name} must be a list or an inclusive start/end range")


def _parse_dataset(raw: Any, index: int) -> DatasetSpec:
    values = _mapping(raw, f"datasets[{index}]")
    allowed = {
        "name",
        "variable",
        "resolution_tag",
        "shape",
        "height",
        "width",
        "timestep_digits",
        "splits",
        "train_timesteps",
        "validation_timesteps",
        "test_timesteps",
        "compressors",
        "even_compressor",
        "odd_compressor",
        "output_subdir",
        "templates",
        "original_template",
        "compressed_template",
        "reconstruction_template",
    }
    _unexpected(f"datasets[{index}]", values, allowed)
    for required in ("name", "variable", "resolution_tag"):
        if required not in values:
            raise ConfigError(f"datasets[{index}].{required} is required")

    shape = values.get("shape")
    if shape is not None:
        if "height" in values or "width" in values:
            raise ConfigError(f"datasets[{index}] cannot combine shape with height/width")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or len(shape) != 2:
            raise ConfigError(f"datasets[{index}].shape must be [height, width]")
        height, width = shape
    else:
        if "height" not in values or "width" not in values:
            raise ConfigError(f"datasets[{index}] requires shape or height and width")
        height, width = values["height"], values["width"]

    splits = _mapping(values.get("splits"), f"datasets[{index}].splits")
    _unexpected(f"datasets[{index}].splits", splits, {"train", "validation", "val", "test"})
    if "validation" in splits and "val" in splits:
        raise ConfigError(f"datasets[{index}].splits cannot define validation and val")
    direct_split_names = {
        "train": "train_timesteps",
        "validation": "validation_timesteps",
        "test": "test_timesteps",
    }
    parsed_splits: dict[str, tuple[int, ...]] = {}
    for split, direct_name in direct_split_names.items():
        split_key = "val" if split == "validation" and "val" in splits else split
        if split_key in splits and direct_name in values:
            raise ConfigError(
                f"datasets[{index}] defines both splits.{split_key} and {direct_name}"
            )
        raw_split = splits.get(split_key, values.get(direct_name))
        parsed_splits[split] = _parse_timesteps(raw_split, f"datasets[{index}].splits.{split}")

    compressors = _mapping(values.get("compressors"), f"datasets[{index}].compressors")
    _unexpected(f"datasets[{index}].compressors", compressors, {"even", "odd"})
    if "even" in compressors and "even_compressor" in values:
        raise ConfigError(f"datasets[{index}] defines the even compressor twice")
    if "odd" in compressors and "odd_compressor" in values:
        raise ConfigError(f"datasets[{index}] defines the odd compressor twice")

    templates = _mapping(values.get("templates"), f"datasets[{index}].templates")
    _unexpected(
        f"datasets[{index}].templates",
        templates,
        {"original", "compressed", "reconstruction"},
    )
    template_fields = {
        "original": "original_template",
        "compressed": "compressed_template",
        "reconstruction": "reconstruction_template",
    }
    parsed_templates: dict[str, str] = {}
    for short_name, field_name in template_fields.items():
        if short_name in templates and field_name in values:
            raise ConfigError(f"datasets[{index}] defines {field_name} twice")
        if short_name in templates:
            parsed_templates[field_name] = templates[short_name]
        elif field_name in values:
            parsed_templates[field_name] = values[field_name]

    return DatasetSpec(
        name=values["name"],
        variable=values["variable"],
        resolution_tag=values["resolution_tag"],
        height=height,
        width=width,
        train_timesteps=parsed_splits["train"],
        validation_timesteps=parsed_splits["validation"],
        test_timesteps=parsed_splits["test"],
        even_compressor=compressors.get("even", values.get("even_compressor", "sz3")),
        odd_compressor=compressors.get("odd", values.get("odd_compressor", "sz3")),
        timestep_digits=values.get("timestep_digits", 4),
        output_subdir=values.get("output_subdir", "{name}/{compressor}"),
        **parsed_templates,
    )


_ConfigSectionT = TypeVar(
    "_ConfigSectionT", ModelConfig, TrainingConfig, EvaluationConfig, TrackingConfig
)


def _parse_dataclass(
    cls: type[_ConfigSectionT],
    raw: Any,
    section: str,
) -> _ConfigSectionT:
    values = _mapping(raw, section)
    allowed = set(cls.__dataclass_fields__)
    _unexpected(section, values, allowed)
    try:
        return cls(**values)
    except TypeError as error:
        raise ConfigError(f"invalid {section} configuration: {error}") from error


def _set_dotted(root: dict[str, Any], dotted_key: str, value: Any) -> None:
    if not dotted_key or dotted_key.startswith(".") or dotted_key.endswith("."):
        raise ConfigError(f"invalid override key {dotted_key!r}")
    parts = dotted_key.split(".")
    current: Any = root
    for position, part in enumerate(parts[:-1]):
        next_part = parts[position + 1]
        if isinstance(current, list):
            try:
                index = int(part)
                current = current[index]
            except (ValueError, IndexError) as error:
                raise ConfigError(f"invalid list index in override {dotted_key!r}") from error
        elif isinstance(current, dict):
            if part not in current:
                current[part] = [] if next_part.isdigit() else {}
            current = current[part]
        else:
            raise ConfigError(f"cannot descend through override {dotted_key!r}")
    final = parts[-1]
    if isinstance(current, list):
        try:
            current[int(final)] = value
        except (ValueError, IndexError) as error:
            raise ConfigError(f"invalid list index in override {dotted_key!r}") from error
    elif isinstance(current, dict):
        current[final] = value
    else:
        raise ConfigError(f"cannot apply override {dotted_key!r}")


def _apply_overrides(
    document: dict[str, Any], overrides: Mapping[str, Any] | Sequence[str] | None
) -> dict[str, Any]:
    result = copy.deepcopy(document)
    if overrides is None:
        return result
    if isinstance(overrides, Mapping):
        items = list(overrides.items())
    elif isinstance(overrides, Sequence) and not isinstance(overrides, (str, bytes)):
        items = []
        for item in overrides:
            if not isinstance(item, str) or "=" not in item:
                raise ConfigError("string overrides must use dotted.path=value")
            key, raw_value = item.split("=", 1)
            items.append((key, yaml.safe_load(raw_value)))
    else:
        raise ConfigError("overrides must be a mapping or a sequence of key=value strings")

    for key, value in items:
        if not isinstance(key, str):
            raise ConfigError("override keys must be strings")
        if "." in key:
            _set_dotted(result, key, value)
        elif isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(
    path: str | os.PathLike[str],
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
) -> ExperimentConfig:
    """Load and validate an experiment YAML file.

    ``overrides`` accepts either a mapping (including dotted keys) or CLI-style
    strings such as ``["training.epochs=20", "tracking.enabled=true"]``.
    Ranges in dataset splits are inclusive at both ends.
    """

    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except OSError as error:
        raise ConfigError(f"cannot read configuration {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {config_path}: {error}") from error
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ConfigError("the YAML document root must be a mapping")
    document = _apply_overrides(dict(loaded), overrides)
    allowed = {
        "name",
        "paths",
        "datasets",
        "compressors",
        "model",
        "training",
        "evaluation",
        "tracking",
    }
    _unexpected("root", document, allowed)

    paths = _mapping(document.get("paths"), "paths")
    _unexpected("paths", paths, {"data_root", "output_root"})
    if "data_root" not in paths:
        raise ConfigError("paths.data_root is required")
    base_dir = config_path.parent
    path_config = PathConfig(
        data_root=_resolve_path(paths["data_root"], base_dir, "data_root"),
        output_root=_resolve_path(paths.get("output_root", "outputs"), base_dir, "output_root"),
    )

    raw_datasets = document.get("datasets")
    if not isinstance(raw_datasets, Sequence) or isinstance(raw_datasets, (str, bytes)):
        raise ConfigError("datasets must be a list")
    datasets = tuple(_parse_dataset(raw, index) for index, raw in enumerate(raw_datasets))

    raw_compressors = _mapping(document.get("compressors"), "compressors")
    if "extensions" in raw_compressors:
        _unexpected("compressors", raw_compressors, {"extensions"})
        raw_extensions = _mapping(raw_compressors["extensions"], "compressors.extensions")
    else:
        raw_extensions = raw_compressors
    extensions = dict(DEFAULT_COMPRESSOR_EXTENSIONS)
    extensions.update(raw_extensions)

    return ExperimentConfig(
        name=document.get("name", "ptunet-experiment"),
        paths=path_config,
        datasets=datasets,
        compressor_extensions=extensions,
        model=_parse_dataclass(ModelConfig, document.get("model"), "model"),
        training=_parse_dataclass(TrainingConfig, document.get("training"), "training"),
        evaluation=_parse_dataclass(EvaluationConfig, document.get("evaluation"), "evaluation"),
        tracking=_parse_dataclass(TrackingConfig, document.get("tracking"), "tracking"),
    )


__all__ = [
    "DEFAULT_COMPRESSOR_EXTENSIONS",
    "ConfigError",
    "DatasetSpec",
    "EvaluationConfig",
    "ExperimentConfig",
    "LossConfig",
    "ModelConfig",
    "PathConfig",
    "TrackingConfig",
    "TrainingConfig",
    "load_config",
]
