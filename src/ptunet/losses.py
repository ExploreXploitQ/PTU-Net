"""Losses used by PTU-Net experiments.

The default weights reproduce the optimization objective in the original
single-file experiment. Spatial-gradient and spectral terms are available for
controlled ablations, but remain disabled unless a configuration enables them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LossWeights:
    """Weights for the composite reconstruction objective."""

    mse: float = 0.7
    charbonnier: float = 0.3
    correction: float = 1.0e-4
    gradient: float = 0.0
    spectral: float = 0.0

    def __post_init__(self) -> None:
        values = (self.mse, self.charbonnier, self.correction, self.gradient, self.spectral)
        if any(value < 0.0 for value in values):
            raise ValueError("Loss weights must be non-negative")
        if self.mse + self.charbonnier + self.gradient + self.spectral <= 0.0:
            raise ValueError("At least one reconstruction loss weight must be positive")


@dataclass
class LossOutput:
    """Differentiable total loss and detached logging components."""

    total: Tensor
    components: dict[str, float]


def _spatial_gradient_loss(prediction: Tensor, target: Tensor) -> Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (
        torch.mean(torch.abs(pred_dx - target_dx)) + torch.mean(torch.abs(pred_dy - target_dy))
    )


def _spectral_amplitude_loss(prediction: Tensor, target: Tensor) -> Tensor:
    pred_spectrum = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_spectrum = torch.fft.rfft2(target.float(), norm="ortho")
    pred_amplitude = torch.log1p(torch.abs(pred_spectrum))
    target_amplitude = torch.log1p(torch.abs(target_spectrum))
    return torch.mean(torch.abs(pred_amplitude - target_amplitude))


class CompositeReconstructionLoss(nn.Module):
    """MSE and robust reconstruction loss with optional structure penalties."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        charbonnier_epsilon: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        if charbonnier_epsilon <= 0.0:
            raise ValueError("charbonnier_epsilon must be positive")
        self.charbonnier_epsilon = float(charbonnier_epsilon)

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        correction: Tensor | None = None,
    ) -> LossOutput:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
            )

        error = prediction - target
        mse = torch.mean(error.square())
        charbonnier = torch.mean(torch.sqrt(error.square() + self.charbonnier_epsilon**2))
        correction_penalty = (
            torch.mean(correction.square()) if correction is not None else prediction.new_zeros(())
        )
        gradient = (
            _spatial_gradient_loss(prediction, target)
            if self.weights.gradient > 0.0
            else prediction.new_zeros(())
        )
        spectral = (
            _spectral_amplitude_loss(prediction, target)
            if self.weights.spectral > 0.0
            else prediction.new_zeros(())
        )
        total = (
            self.weights.mse * mse
            + self.weights.charbonnier * charbonnier
            + self.weights.correction * correction_penalty
            + self.weights.gradient * gradient
            + self.weights.spectral * spectral
        )
        components = {
            "loss": float(total.detach()),
            "mse": float(mse.detach()),
            "charbonnier": float(charbonnier.detach()),
            "correction": float(correction_penalty.detach()),
            "gradient": float(gradient.detach()),
            "spectral": float(spectral.detach()),
        }
        return LossOutput(total=total, components=components)


__all__ = ["CompositeReconstructionLoss", "LossOutput", "LossWeights"]
