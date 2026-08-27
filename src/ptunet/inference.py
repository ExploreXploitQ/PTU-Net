"""Patchwise full-field reconstruction with weighted overlap-add blending."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ptunet.data import (
    NormalizationStats,
    center_difference_features,
    compute_patch_positions,
    normalized_raw_features,
)
from ptunet.engine import resolve_device


@dataclass(frozen=True)
class ReconstructionResult:
    """A reconstructed field and its measured inference cost."""

    field: np.ndarray
    elapsed_seconds: float
    patch_count: int


class TemporalInferencePatches(Dataset[Tensor]):
    """Lazy patches from three already reconstructed temporal fields."""

    def __init__(
        self,
        previous: np.ndarray,
        center: np.ndarray,
        next_field: np.ndarray,
        normalization: NormalizationStats,
        *,
        patch_size: int,
        stride: int,
        input_mode: Literal["center_difference", "raw_frames"] = "center_difference",
    ) -> None:
        fields = tuple(np.asarray(field) for field in (previous, center, next_field))
        if any(field.ndim != 2 for field in fields):
            raise ValueError("inference fields must be two-dimensional")
        if any(field.shape != fields[0].shape for field in fields[1:]):
            raise ValueError("inference fields must have identical shapes")
        if input_mode not in {"center_difference", "raw_frames"}:
            raise ValueError("input_mode must be center_difference or raw_frames")
        self.fields = fields
        self.normalization = normalization
        self.patch_size = patch_size
        self.input_mode = input_mode
        self.positions = compute_patch_positions(fields[0].shape, patch_size, stride)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> Tensor:
        row, column = self.positions[index]
        row_slice = slice(row, row + self.patch_size)
        column_slice = slice(column, column + self.patch_size)
        patches = tuple(field[row_slice, column_slice] for field in self.fields)
        if self.input_mode == "center_difference":
            inputs = center_difference_features(
                patches[0], patches[1], patches[2], self.normalization
            )
        else:
            inputs = normalized_raw_features(patches[0], patches[1], patches[2], self.normalization)
        return torch.from_numpy(inputs)


def gaussian_blend_window(size: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """Create a positive, unit-peak 2D Gaussian overlap-add window."""

    if size < 1:
        raise ValueError("window size must be positive")
    if sigma_ratio <= 0.0:
        raise ValueError("sigma_ratio must be positive")
    axis = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size, dtype=np.float64)
    epsilon = float(np.finfo(np.float64).eps)
    sigma = max(float(size) * float(sigma_ratio), epsilon)
    one_dimensional = np.exp(-0.5 * (axis / sigma) ** 2)
    window = np.outer(one_dimensional, one_dimensional)
    peak = float(np.max(window))
    return np.asarray(window / peak, dtype=np.float32)


@torch.inference_mode()
def reconstruct_field(
    model: nn.Module,
    temporal_fields: Sequence[np.ndarray],
    normalization: NormalizationStats,
    *,
    patch_size: int,
    stride: int,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | torch.device = "auto",
    input_mode: Literal["center_difference", "raw_frames"] = "center_difference",
    sigma_ratio: float = 0.25,
) -> ReconstructionResult:
    """Reconstruct one center field from previous, center, and next inputs."""

    if len(temporal_fields) != 3:
        raise ValueError("temporal_fields must contain previous, center, and next arrays")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    dataset = TemporalInferencePatches(
        temporal_fields[0],
        temporal_fields[1],
        temporal_fields[2],
        normalization,
        patch_size=patch_size,
        stride=stride,
        input_mode=input_mode,
    )
    resolved_device = resolve_device(device) if isinstance(device, str) else device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model = model.to(resolved_device).eval()
    height, width = dataset.fields[0].shape
    accumulated = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    window = gaussian_blend_window(patch_size, sigma_ratio)

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    started = time.perf_counter()
    position_index = 0
    for inputs in loader:
        predictions = model(inputs.to(resolved_device, non_blocking=True)).float().cpu().numpy()
        for patch in predictions:
            row, column = dataset.positions[position_index]
            accumulated[row : row + patch_size, column : column + patch_size] += patch * window
            weights[row : row + patch_size, column : column + patch_size] += window
            position_index += 1
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    elapsed = time.perf_counter() - started

    if position_index != len(dataset):
        raise RuntimeError(f"inference produced {position_index} patches; expected {len(dataset)}")
    if np.any(weights <= 0.0):
        raise RuntimeError("overlap-add patch geometry left uncovered pixels")
    normalized = np.asarray(accumulated / weights, dtype=np.float32)
    reconstructed = normalization.denormalize(normalized)
    return ReconstructionResult(reconstructed, elapsed, len(dataset))


__all__ = [
    "ReconstructionResult",
    "TemporalInferencePatches",
    "gaussian_blend_window",
    "reconstruct_field",
]
