from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ptunet.data import NormalizationStats
from ptunet.inference import (
    TemporalInferencePatches,
    gaussian_blend_window,
    reconstruct_field,
)


class CenterIdentity(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, 0]


def test_overlap_add_reconstructs_center_identity() -> None:
    rng = np.random.default_rng(9)
    center = rng.normal(size=(13, 17)).astype(np.float32)
    previous = center - 0.25
    next_field = center + 0.5
    stats = NormalizationStats(
        mean=float(center.mean()),
        std=float(center.std()),
        count=center.size,
        minimum=float(center.min()),
        maximum=float(center.max()),
    )

    result = reconstruct_field(
        CenterIdentity(),
        (previous, center, next_field),
        stats,
        patch_size=8,
        stride=5,
        batch_size=3,
        device="cpu",
    )

    np.testing.assert_allclose(result.field, center, rtol=1.0e-6, atol=1.0e-6)
    assert result.patch_count > 1
    assert result.elapsed_seconds >= 0.0


def test_gaussian_window_is_positive_and_symmetric() -> None:
    window = gaussian_blend_window(8)

    assert np.all(window > 0)
    np.testing.assert_array_equal(window, window[::-1, :])
    np.testing.assert_array_equal(window, window[:, ::-1])
    assert window.max() == 1.0


@pytest.mark.parametrize("non_finite", [np.nan, np.inf])
def test_raw_frame_inference_rejects_non_finite_values(non_finite: float) -> None:
    previous = np.zeros((4, 4), dtype=np.float32)
    center = np.ones((4, 4), dtype=np.float32)
    next_field = np.full((4, 4), 2.0, dtype=np.float32)
    previous[0, 0] = non_finite
    stats = NormalizationStats(mean=1.0, std=0.5, count=16, minimum=0.0, maximum=2.0)
    dataset = TemporalInferencePatches(
        previous,
        center,
        next_field,
        stats,
        patch_size=4,
        stride=4,
        input_mode="raw_frames",
    )

    with pytest.raises(ValueError, match="non-finite"):
        dataset[0]
