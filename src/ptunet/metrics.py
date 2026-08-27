"""Streaming full-field regression metrics for scientific reconstruction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    """Aggregate metrics; SSIM is global rather than windowed/image SSIM."""

    count: int
    mse: float
    mae: float
    psnr: float
    global_ssim: float
    data_range: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "mse": self.mse,
            "mae": self.mae,
            "psnr": self.psnr,
            "global_ssim": self.global_ssim,
            "data_range": self.data_range,
        }


def _validated_pair(prediction: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = np.asarray(prediction)
    reference_array = np.asarray(reference)
    if prediction_array.shape != reference_array.shape:
        raise ValueError(
            "prediction and reference shapes differ: "
            f"{prediction_array.shape} != {reference_array.shape}"
        )
    if prediction_array.size == 0:
        raise ValueError("metrics require at least one value")
    if not np.issubdtype(prediction_array.dtype, np.number):
        raise TypeError("prediction must contain numeric values")
    if not np.issubdtype(reference_array.dtype, np.number):
        raise TypeError("reference must contain numeric values")
    if np.iscomplexobj(prediction_array) or np.iscomplexobj(reference_array):
        raise TypeError("prediction and reference must contain real values")
    return prediction_array, reference_array


class StreamingRegressionMetrics:
    """Accumulate errors and global moments without materializing full copies."""

    def __init__(self, *, chunk_elements: int = 1_048_576) -> None:
        if (
            isinstance(chunk_elements, bool)
            or not isinstance(chunk_elements, int)
            or chunk_elements <= 0
        ):
            raise ValueError("chunk_elements must be a positive integer")
        self.chunk_elements = chunk_elements
        self.count = 0
        self.squared_error_sum = 0.0
        self.absolute_error_sum = 0.0
        self.prediction_mean = 0.0
        self.reference_mean = 0.0
        self.prediction_m2 = 0.0
        self.reference_m2 = 0.0
        self.cross_moment = 0.0
        self.reference_minimum = math.inf
        self.reference_maximum = -math.inf

    def update(self, prediction: np.ndarray, reference: np.ndarray) -> None:
        prediction_array, reference_array = _validated_pair(prediction, reference)
        prediction_flat = prediction_array.reshape(-1)
        reference_flat = reference_array.reshape(-1)
        for start in range(0, prediction_flat.size, self.chunk_elements):
            stop = start + self.chunk_elements
            pred = np.asarray(prediction_flat[start:stop], dtype=np.float64)
            ref = np.asarray(reference_flat[start:stop], dtype=np.float64)
            if not np.isfinite(pred).all() or not np.isfinite(ref).all():
                raise ValueError("prediction and reference must contain only finite values")

            error = pred - ref
            self.squared_error_sum += float(np.dot(error, error))
            self.absolute_error_sum += float(np.sum(np.abs(error), dtype=np.float64))
            self.reference_minimum = min(self.reference_minimum, float(np.min(ref)))
            self.reference_maximum = max(self.reference_maximum, float(np.max(ref)))

            chunk_count = int(pred.size)
            prediction_mean = float(np.mean(pred, dtype=np.float64))
            reference_mean = float(np.mean(ref, dtype=np.float64))
            prediction_centered = pred - prediction_mean
            reference_centered = ref - reference_mean
            prediction_m2 = float(np.dot(prediction_centered, prediction_centered))
            reference_m2 = float(np.dot(reference_centered, reference_centered))
            cross_moment = float(np.dot(prediction_centered, reference_centered))
            self._merge_moments(
                chunk_count,
                prediction_mean,
                reference_mean,
                prediction_m2,
                reference_m2,
                cross_moment,
            )

    def _merge_moments(
        self,
        count: int,
        prediction_mean: float,
        reference_mean: float,
        prediction_m2: float,
        reference_m2: float,
        cross_moment: float,
    ) -> None:
        if self.count == 0:
            self.count = count
            self.prediction_mean = prediction_mean
            self.reference_mean = reference_mean
            self.prediction_m2 = prediction_m2
            self.reference_m2 = reference_m2
            self.cross_moment = cross_moment
            return
        total = self.count + count
        prediction_delta = prediction_mean - self.prediction_mean
        reference_delta = reference_mean - self.reference_mean
        correction = self.count * count / total
        self.prediction_m2 += prediction_m2 + prediction_delta**2 * correction
        self.reference_m2 += reference_m2 + reference_delta**2 * correction
        self.cross_moment += cross_moment + prediction_delta * reference_delta * correction
        self.prediction_mean += prediction_delta * count / total
        self.reference_mean += reference_delta * count / total
        self.count = total

    def merge(self, other: StreamingRegressionMetrics) -> None:
        """Merge an independently accumulated shard."""

        if other.count == 0:
            return
        self.squared_error_sum += other.squared_error_sum
        self.absolute_error_sum += other.absolute_error_sum
        self.reference_minimum = min(self.reference_minimum, other.reference_minimum)
        self.reference_maximum = max(self.reference_maximum, other.reference_maximum)
        self._merge_moments(
            other.count,
            other.prediction_mean,
            other.reference_mean,
            other.prediction_m2,
            other.reference_m2,
            other.cross_moment,
        )

    def _require_values(self) -> None:
        if self.count == 0:
            raise ValueError("cannot compute metrics before any values are added")

    @property
    def mean_squared_error(self) -> float:
        self._require_values()
        return self.squared_error_sum / self.count

    @property
    def mean_absolute_error(self) -> float:
        self._require_values()
        return self.absolute_error_sum / self.count

    def _data_range(self, data_range: float | None) -> float:
        self._require_values()
        if data_range is not None:
            if (
                isinstance(data_range, bool)
                or not isinstance(data_range, (int, float))
                or not math.isfinite(data_range)
                or data_range <= 0
            ):
                raise ValueError("data_range must be a finite positive number")
            return float(data_range)
        inferred = self.reference_maximum - self.reference_minimum
        if inferred == 0 and self.mean_squared_error != 0:
            raise ValueError(
                "reference is constant; provide a positive data_range for non-perfect results"
            )
        return inferred

    def peak_signal_to_noise_ratio(self, *, data_range: float | None = None) -> float:
        error = self.mean_squared_error
        resolved_range = self._data_range(data_range)
        if error == 0:
            return math.inf
        return 10.0 * math.log10(resolved_range * resolved_range / error)

    def global_structural_similarity(
        self,
        *,
        data_range: float | None = None,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> float:
        """Compute one SSIM value from population moments over the whole field."""

        if k1 <= 0 or k2 <= 0:
            raise ValueError("SSIM constants k1 and k2 must be positive")
        resolved_range = self._data_range(data_range)
        if resolved_range == 0:
            return 1.0  # Only reachable for identical constant arrays.
        prediction_variance = max(0.0, self.prediction_m2 / self.count)
        reference_variance = max(0.0, self.reference_m2 / self.count)
        covariance = self.cross_moment / self.count
        c1 = (k1 * resolved_range) ** 2
        c2 = (k2 * resolved_range) ** 2
        numerator = (2 * self.prediction_mean * self.reference_mean + c1) * (2 * covariance + c2)
        denominator = (self.prediction_mean**2 + self.reference_mean**2 + c1) * (
            prediction_variance + reference_variance + c2
        )
        if denominator == 0:
            return 1.0 if self.mean_squared_error == 0 else 0.0
        return float(np.clip(numerator / denominator, -1.0, 1.0))

    def finalize(
        self,
        *,
        data_range: float | None = None,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> RegressionMetrics:
        resolved_range = self._data_range(data_range)
        return RegressionMetrics(
            count=self.count,
            mse=self.mean_squared_error,
            mae=self.mean_absolute_error,
            psnr=self.peak_signal_to_noise_ratio(data_range=resolved_range or None),
            global_ssim=self.global_structural_similarity(
                data_range=resolved_range or None, k1=k1, k2=k2
            ),
            data_range=resolved_range,
        )


def _accumulate(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_elements: int = 1_048_576,
) -> StreamingRegressionMetrics:
    accumulator = StreamingRegressionMetrics(chunk_elements=chunk_elements)
    accumulator.update(prediction, reference)
    return accumulator


def mse(prediction: np.ndarray, reference: np.ndarray) -> float:
    return _accumulate(prediction, reference).mean_squared_error


def mae(prediction: np.ndarray, reference: np.ndarray) -> float:
    return _accumulate(prediction, reference).mean_absolute_error


def psnr(
    prediction: np.ndarray, reference: np.ndarray, *, data_range: float | None = None
) -> float:
    """Compute PSNR using an explicit range or ``max(reference)-min(reference)``."""

    return _accumulate(prediction, reference).peak_signal_to_noise_ratio(data_range=data_range)


def global_ssim(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    data_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute global, population-moment SSIM (not local/windowed image SSIM)."""

    return _accumulate(prediction, reference).global_structural_similarity(
        data_range=data_range, k1=k1, k2=k2
    )


def compute_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    data_range: float | None = None,
    chunk_elements: int = 1_048_576,
) -> RegressionMetrics:
    """Compute all supported metrics in one streaming pass."""

    return _accumulate(prediction, reference, chunk_elements=chunk_elements).finalize(
        data_range=data_range
    )


__all__ = [
    "RegressionMetrics",
    "StreamingRegressionMetrics",
    "compute_metrics",
    "global_ssim",
    "mae",
    "mse",
    "psnr",
]
