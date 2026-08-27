"""Lazy temporal-field datasets, normalization, and patch geometry."""

from __future__ import annotations

import bisect
import math
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ptunet.config import DatasetSpec, ExperimentConfig
from ptunet.io import (
    CompressorResolver,
    compressed_path,
    open_float32_memmap,
    original_path,
    validate_float32_file,
)

MODEL_INPUT_CHANNELS = ("center", "previous_minus_center", "next_minus_center")
RAW_INPUT_CHANNELS = ("previous", "center", "next")


@dataclass(frozen=True)
class NormalizationStats:
    """Population statistics fitted exclusively on original training fields."""

    mean: float
    std: float
    count: int
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("normalization count must be a positive integer")
        values = (self.mean, self.std, self.minimum, self.maximum)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("normalization statistics must be finite")
        if self.std <= 0:
            raise ValueError("normalization standard deviation must be positive")
        if self.minimum > self.maximum:
            raise ValueError("normalization minimum cannot exceed maximum")

    @property
    def data_range(self) -> float:
        return self.maximum - self.minimum

    def normalize(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return np.asarray((array - np.float32(self.mean)) / np.float32(self.std), dtype=np.float32)

    def denormalize(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return np.asarray(array * np.float32(self.std) + np.float32(self.mean), dtype=np.float32)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "std": self.std,
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, float | int]) -> NormalizationStats:
        minimum = values.get("minimum", values.get("min"))
        maximum = values.get("maximum", values.get("max"))
        if minimum is None or maximum is None:
            raise ValueError("normalization mapping requires minimum and maximum")
        return cls(
            mean=float(values["mean"]),
            std=float(values["std"]),
            count=int(values["count"]),
            minimum=float(minimum),
            maximum=float(maximum),
        )


class StreamingStats:
    """Numerically stable, mergeable population statistics."""

    def __init__(self, *, chunk_elements: int = 1_048_576) -> None:
        if (
            isinstance(chunk_elements, bool)
            or not isinstance(chunk_elements, int)
            or chunk_elements <= 0
        ):
            raise ValueError("chunk_elements must be a positive integer")
        self.chunk_elements = chunk_elements
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: np.ndarray | Sequence[float]) -> None:
        source = np.asarray(values)
        flat = source.reshape(-1)
        for start in range(0, flat.size, self.chunk_elements):
            chunk = np.asarray(flat[start : start + self.chunk_elements], dtype=np.float64)
            if chunk.size == 0:
                continue
            if not np.isfinite(chunk).all():
                raise ValueError("cannot fit normalization statistics to non-finite values")
            chunk_count = int(chunk.size)
            chunk_mean = float(np.mean(chunk, dtype=np.float64))
            centered = chunk - chunk_mean
            chunk_m2 = float(np.dot(centered, centered))

            if self.count == 0:
                self.count = chunk_count
                self.mean = chunk_mean
                self.m2 = chunk_m2
            else:
                combined_count = self.count + chunk_count
                delta = chunk_mean - self.mean
                self.m2 += chunk_m2 + delta * delta * self.count * chunk_count / combined_count
                self.mean += delta * chunk_count / combined_count
                self.count = combined_count
            self.minimum = min(self.minimum, float(np.min(chunk)))
            self.maximum = max(self.maximum, float(np.max(chunk)))

    def merge(self, other: StreamingStats) -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.minimum = other.minimum
            self.maximum = other.maximum
            return
        combined_count = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / combined_count
        self.mean += delta * other.count / combined_count
        self.count = combined_count
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)

    def finalize(self) -> NormalizationStats:
        if self.count == 0:
            raise ValueError("cannot finalize empty normalization statistics")
        variance = max(0.0, self.m2 / self.count)
        std = math.sqrt(variance)
        if std == 0:
            raise ValueError("training fields are constant; standard deviation is zero")
        return NormalizationStats(
            mean=self.mean,
            std=std,
            count=self.count,
            minimum=self.minimum,
            maximum=self.maximum,
        )


def compute_normalization_stats(
    arrays: Iterable[np.ndarray], *, chunk_elements: int = 1_048_576
) -> NormalizationStats:
    """Fit population statistics without concatenating fields in memory."""

    accumulator = StreamingStats(chunk_elements=chunk_elements)
    for array in arrays:
        accumulator.update(array)
    return accumulator.finalize()


def compute_training_normalization(
    specs: Sequence[DatasetSpec],
    data_root: str | os.PathLike[str],
    *,
    chunk_elements: int = 1_048_576,
) -> NormalizationStats:
    """Fit one normalization from original train targets across all datasets."""

    accumulator = StreamingStats(chunk_elements=chunk_elements)
    for spec in specs:
        for timestep in spec.train_timesteps:
            field = open_float32_memmap(original_path(spec, data_root, timestep), spec.shape)
            accumulator.update(field)
            del field
    return accumulator.finalize()


@dataclass(frozen=True)
class TemporalWindow:
    """Semantic temporal inputs; ``center`` is never inferred from tuple position."""

    previous: int
    center: int
    next: int

    @property
    def timesteps(self) -> tuple[int, int, int]:
        return (self.previous, self.center, self.next)


def resolve_temporal_window(center: int, available_timesteps: Iterable[int]) -> TemporalWindow:
    """Choose a robust three-frame window while keeping the center explicit.

    At a lower boundary with two future frames, the semantic layout is
    ``(+2, center, +1)``; at an upper boundary it is ``(-1, center, -2)``.
    With only one side frame, the unavailable side repeats the center and
    therefore contributes a zero difference feature.
    """

    available = sorted(set(available_timesteps))
    if center not in available:
        raise ValueError(f"center timestep {center} is not available")
    lower = [value for value in available if value < center]
    upper = [value for value in available if value > center]
    nearest_lower = lower[-1] if lower else None
    nearest_upper = upper[0] if upper else None

    if nearest_lower is not None and nearest_upper is not None:
        return TemporalWindow(nearest_lower, center, nearest_upper)
    if nearest_lower is None and len(upper) >= 2:
        return TemporalWindow(upper[1], center, upper[0])
    if nearest_upper is None and len(lower) >= 2:
        return TemporalWindow(lower[-1], center, lower[-2])
    if nearest_lower is None and nearest_upper is not None:
        return TemporalWindow(center, center, nearest_upper)
    if nearest_lower is not None and nearest_upper is None:
        return TemporalWindow(nearest_lower, center, center)
    return TemporalWindow(center, center, center)


def resolve_temporal_window_from_files(
    spec: DatasetSpec,
    center: int,
    data_root: str | os.PathLike[str],
    resolver: CompressorResolver,
    *,
    search_radius: int = 2,
) -> TemporalWindow:
    """Resolve a window from compressed files that actually exist on disk."""

    if isinstance(search_radius, bool) or not isinstance(search_radius, int) or search_radius <= 0:
        raise ValueError("search_radius must be a positive integer")
    candidates = range(max(0, center - search_radius), center + search_radius + 1)
    available = [
        timestep
        for timestep in candidates
        if compressed_path(spec, data_root, timestep, resolver).is_file()
    ]
    if center not in available:
        path = compressed_path(spec, data_root, center, resolver)
        raise FileNotFoundError(f"compressed center field does not exist: {path}")
    return resolve_temporal_window(center, available)


def center_difference_features(
    previous: np.ndarray,
    center: np.ndarray,
    next_field: np.ndarray,
    normalization: NormalizationStats,
) -> np.ndarray:
    """Return ``[normalized center, (prev-center)/std, (next-center)/std]``."""

    previous_array = np.asarray(previous, dtype=np.float32)
    center_array = np.asarray(center, dtype=np.float32)
    next_array = np.asarray(next_field, dtype=np.float32)
    if previous_array.shape != center_array.shape or next_array.shape != center_array.shape:
        raise ValueError("previous, center, and next fields must have identical shapes")
    if not (
        np.isfinite(previous_array).all()
        and np.isfinite(center_array).all()
        and np.isfinite(next_array).all()
    ):
        raise ValueError("temporal fields contain non-finite values")
    scale = np.float32(normalization.std)
    channels = (
        normalization.normalize(center_array),
        np.asarray((previous_array - center_array) / scale, dtype=np.float32),
        np.asarray((next_array - center_array) / scale, dtype=np.float32),
    )
    return np.ascontiguousarray(np.stack(channels, axis=0), dtype=np.float32)


def normalized_raw_features(
    previous: np.ndarray,
    center: np.ndarray,
    next_field: np.ndarray,
    normalization: NormalizationStats,
) -> np.ndarray:
    """Return normalized raw frames in ``[previous, center, next]`` order."""

    arrays = tuple(np.asarray(field, dtype=np.float32) for field in (previous, center, next_field))
    if arrays[0].shape != arrays[1].shape or arrays[2].shape != arrays[1].shape:
        raise ValueError("previous, center, and next fields must have identical shapes")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("temporal fields contain non-finite values")
    return np.ascontiguousarray(
        np.stack(tuple(normalization.normalize(array) for array in arrays), axis=0),
        dtype=np.float32,
    )


def _pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        result = (value[0], value[1])
    else:
        raise ValueError(f"{name} must be an integer or a pair")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def axis_starts(length: int, patch_size: int, stride: int) -> tuple[int, ...]:
    """Return starts that cover an axis, including a final edge-aligned patch."""

    for name, value in (("length", length), ("patch_size", patch_size), ("stride", stride)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if patch_size > length:
        raise ValueError(f"patch_size {patch_size} exceeds axis length {length}")
    if stride > patch_size:
        raise ValueError("stride cannot exceed patch_size because it would leave gaps")
    final_start = length - patch_size
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def compute_patch_positions(
    shape: int | Sequence[int],
    patch_size: int | Sequence[int],
    stride: int | Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """Return row-major patch starts with complete spatial coverage."""

    height, width = _pair(shape, "shape")
    patch_height, patch_width = _pair(patch_size, "patch_size")
    row_stride, column_stride = _pair(stride, "stride")
    rows = axis_starts(height, patch_height, row_stride)
    columns = axis_starts(width, patch_width, column_stride)
    return tuple((row, column) for row in rows for column in columns)


def patch_coverage_counts(
    shape: int | Sequence[int],
    patch_size: int | Sequence[int],
    stride: int | Sequence[int],
) -> np.ndarray:
    """Count how many configured patches cover every pixel."""

    height, width = _pair(shape, "shape")
    patch_height, patch_width = _pair(patch_size, "patch_size")
    coverage = np.zeros((height, width), dtype=np.int32)
    for row, column in compute_patch_positions(shape, patch_size, stride):
        coverage[row : row + patch_height, column : column + patch_width] += 1
    if not np.all(coverage > 0):  # Defensive assertion for future geometry changes.
        raise RuntimeError("patch geometry did not cover every pixel")
    return coverage


@dataclass(frozen=True)
class _FieldPatches:
    spec_index: int
    window: TemporalWindow
    rows: tuple[int, ...]
    columns: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.rows) * len(self.columns)


@dataclass(frozen=True)
class PatchMetadata:
    dataset: str
    center_timestep: int
    temporal_timesteps: tuple[int, int, int]
    row: int
    column: int


class UnifiedPatchDataset(Dataset[tuple[Tensor, Tensor]]):
    """Patch dataset backed by per-worker, lazily opened float32 memmaps."""

    def __init__(
        self,
        specs: Sequence[DatasetSpec],
        data_root: str | os.PathLike[str],
        resolver: CompressorResolver,
        *,
        split: str,
        patch_size: int = 32,
        stride: int = 16,
        normalization: NormalizationStats | None = None,
        temporal_search_radius: int = 2,
        normalization_chunk_elements: int = 1_048_576,
        input_mode: str = "center_difference",
        memmap_cache_size: int = 16,
    ) -> None:
        self.specs = tuple(specs)
        if not self.specs:
            raise ValueError("at least one dataset spec is required")
        self.data_root = Path(data_root).expanduser().resolve()
        self.resolver = resolver
        self.split = "validation" if split.lower() == "val" else split.lower()
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.patch_size = patch_size
        self.stride = stride
        if input_mode not in {"center_difference", "raw_frames"}:
            raise ValueError("input_mode must be center_difference or raw_frames")
        self.input_mode = input_mode
        if (
            isinstance(memmap_cache_size, bool)
            or not isinstance(memmap_cache_size, int)
            or memmap_cache_size <= 0
        ):
            raise ValueError("memmap_cache_size must be a positive integer")
        self.memmap_cache_size = memmap_cache_size

        if normalization is None:
            if self.split != "train":
                raise ValueError(
                    "validation/test datasets require training normalization statistics"
                )
            normalization = compute_training_normalization(
                self.specs,
                self.data_root,
                chunk_elements=normalization_chunk_elements,
            )
        self.normalization = normalization

        fields: list[_FieldPatches] = []
        cumulative: list[int] = []
        sample_count = 0
        for spec_index, spec in enumerate(self.specs):
            rows = axis_starts(spec.height, patch_size, stride)
            columns = axis_starts(spec.width, patch_size, stride)
            for center in spec.timesteps_for_split(self.split):
                target_path = original_path(spec, self.data_root, center)
                validate_float32_file(target_path, spec.shape)
                window = resolve_temporal_window_from_files(
                    spec,
                    center,
                    self.data_root,
                    resolver,
                    search_radius=temporal_search_radius,
                )
                for timestep in set(window.timesteps):
                    validate_float32_file(
                        compressed_path(spec, self.data_root, timestep, resolver), spec.shape
                    )
                field = _FieldPatches(spec_index, window, rows, columns)
                fields.append(field)
                sample_count += field.count
                cumulative.append(sample_count)
        self._fields = tuple(fields)
        self._cumulative = tuple(cumulative)
        self._mmap_cache: OrderedDict[Path, np.memmap] = OrderedDict()
        self._cache_process = os.getpid()

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def _location(self, index: int) -> tuple[_FieldPatches, int, int]:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(f"patch index {index} is out of range for length {length}")
        field_index = bisect.bisect_right(self._cumulative, index)
        previous_total = 0 if field_index == 0 else self._cumulative[field_index - 1]
        within_field = index - previous_total
        field = self._fields[field_index]
        row_index, column_index = divmod(within_field, len(field.columns))
        return field, field.rows[row_index], field.columns[column_index]

    def _memmap(self, path: Path, shape: tuple[int, int]) -> np.memmap:
        process = os.getpid()
        if process != self._cache_process:
            self._mmap_cache = OrderedDict()
            self._cache_process = process
        if path in self._mmap_cache:
            self._mmap_cache.move_to_end(path)
            return self._mmap_cache[path]
        field = open_float32_memmap(path, shape)
        self._mmap_cache[path] = field
        if len(self._mmap_cache) > self.memmap_cache_size:
            _, evicted = self._mmap_cache.popitem(last=False)
            memory_map = getattr(evicted, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        return field

    def sample_metadata(self, index: int) -> PatchMetadata:
        field, row, column = self._location(index)
        spec = self.specs[field.spec_index]
        return PatchMetadata(
            dataset=spec.name,
            center_timestep=field.window.center,
            temporal_timesteps=field.window.timesteps,
            row=row,
            column=column,
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        field, row, column = self._location(index)
        spec = self.specs[field.spec_index]
        row_slice = slice(row, row + self.patch_size)
        column_slice = slice(column, column + self.patch_size)

        temporal: list[np.ndarray] = []
        for timestep in field.window.timesteps:
            path = compressed_path(spec, self.data_root, timestep, self.resolver)
            array = self._memmap(path, spec.shape)
            temporal.append(array[row_slice, column_slice])
        feature_transform = (
            center_difference_features
            if self.input_mode == "center_difference"
            else normalized_raw_features
        )
        inputs = feature_transform(temporal[0], temporal[1], temporal[2], self.normalization)

        target_path = original_path(spec, self.data_root, field.window.center)
        target_field = self._memmap(target_path, spec.shape)
        target_patch = target_field[row_slice, column_slice]
        if not np.isfinite(target_patch).all():
            raise ValueError(f"target field contains non-finite values: {target_path}")
        target = np.ascontiguousarray(
            self.normalization.normalize(target_patch),
            dtype=np.float32,
        )
        return torch.from_numpy(inputs), torch.from_numpy(target)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_mmap_cache"] = OrderedDict()
        state["_cache_process"] = -1
        return state


@dataclass(frozen=True)
class DatasetBundle:
    train: UnifiedPatchDataset
    validation: UnifiedPatchDataset | None
    test: UnifiedPatchDataset | None
    normalization: NormalizationStats

    @property
    def val(self) -> UnifiedPatchDataset | None:
        return self.validation


def build_dataset_splits(
    config: ExperimentConfig,
    *,
    normalization: NormalizationStats | None = None,
    include_validation: bool = True,
    include_test: bool = True,
) -> DatasetBundle:
    """Build requested splits and reuse one exact training normalization.

    Passing ``normalization`` is required when continuing from a checkpoint:
    the tensors seen after resume must use the saved transform rather than a
    newly fitted transform from whichever files happen to be present.
    """

    resolver = CompressorResolver(config.compressor_extensions)
    common: dict[str, Any] = {
        "specs": config.datasets,
        "data_root": config.paths.data_root,
        "resolver": resolver,
        "patch_size": config.model.patch_size,
        "stride": config.model.stride,
        "input_mode": config.model.input_mode,
        "temporal_search_radius": config.evaluation.temporal_search_radius,
    }
    train = UnifiedPatchDataset(
        split="train",
        normalization=normalization,
        normalization_chunk_elements=config.training.normalization_chunk_elements,
        **common,
    )
    validation = (
        UnifiedPatchDataset(split="validation", normalization=train.normalization, **common)
        if include_validation and any(spec.validation_timesteps for spec in config.datasets)
        else None
    )
    test = (
        UnifiedPatchDataset(split="test", normalization=train.normalization, **common)
        if include_test and any(spec.test_timesteps for spec in config.datasets)
        else None
    )
    return DatasetBundle(train, validation, test, train.normalization)


__all__ = [
    "MODEL_INPUT_CHANNELS",
    "RAW_INPUT_CHANNELS",
    "DatasetBundle",
    "NormalizationStats",
    "PatchMetadata",
    "StreamingStats",
    "TemporalWindow",
    "UnifiedPatchDataset",
    "axis_starts",
    "build_dataset_splits",
    "center_difference_features",
    "compute_normalization_stats",
    "compute_patch_positions",
    "compute_training_normalization",
    "normalized_raw_features",
    "patch_coverage_counts",
    "resolve_temporal_window",
    "resolve_temporal_window_from_files",
]
