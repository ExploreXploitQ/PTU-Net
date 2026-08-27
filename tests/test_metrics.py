from __future__ import annotations

import math

import numpy as np
import pytest

from ptunet.metrics import (
    StreamingRegressionMetrics,
    compute_metrics,
    global_ssim,
    mae,
    mse,
    psnr,
)


def test_mse_and_mae_match_known_values() -> None:
    reference = np.array([0.0, 1.0, 2.0])
    prediction = np.array([0.0, 2.0, 4.0])

    assert mse(prediction, reference) == pytest.approx(5.0 / 3.0)
    assert mae(prediction, reference) == pytest.approx(1.0)


def test_perfect_reconstruction_has_infinite_psnr_and_unit_ssim() -> None:
    reference = np.array([2.0, 2.0, 2.0], dtype=np.float32)

    result = compute_metrics(reference, reference)

    assert result.mse == 0
    assert math.isinf(result.psnr)
    assert result.global_ssim == 1
    assert result.data_range == 0


def test_psnr_infers_reference_range_not_maximum_absolute_value() -> None:
    reference = np.array([-1.0, 1.0])
    prediction = np.array([0.0, 0.0])

    assert psnr(prediction, reference) == pytest.approx(10 * math.log10(4.0))


def test_constant_reference_requires_explicit_range_when_imperfect() -> None:
    reference = np.ones(4)
    prediction = np.zeros(4)

    with pytest.raises(ValueError, match="constant"):
        compute_metrics(prediction, reference)

    result = compute_metrics(prediction, reference, data_range=2.0)
    assert result.mse == 1.0
    assert result.psnr == pytest.approx(10 * math.log10(4.0))
    assert -1 <= result.global_ssim <= 1


def test_streaming_updates_and_merges_match_single_pass() -> None:
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(11, 13)).astype(np.float32)
    prediction = reference + rng.normal(scale=0.1, size=reference.shape).astype(np.float32)
    expected = compute_metrics(prediction, reference, chunk_elements=17)
    first = StreamingRegressionMetrics(chunk_elements=9)
    second = StreamingRegressionMetrics(chunk_elements=7)
    first.update(prediction[:5], reference[:5])
    second.update(prediction[5:], reference[5:])

    first.merge(second)
    actual = first.finalize()

    assert actual.count == expected.count
    assert actual.mse == pytest.approx(expected.mse, rel=1e-14)
    assert actual.mae == pytest.approx(expected.mae, rel=1e-14)
    assert actual.psnr == pytest.approx(expected.psnr, rel=1e-14)
    assert actual.global_ssim == pytest.approx(expected.global_ssim, rel=1e-14)


def test_global_ssim_is_symmetric_for_a_fixed_range() -> None:
    left = np.array([0.0, 0.5, 1.0, 1.5])
    right = np.array([0.1, 0.4, 0.8, 1.6])

    assert global_ssim(left, right, data_range=2.0) == pytest.approx(
        global_ssim(right, left, data_range=2.0)
    )


def test_invalid_shapes_and_nonfinite_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        mse(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        mae(np.array([np.nan]), np.array([0.0]))
