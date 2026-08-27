from __future__ import annotations

import pytest
import torch

from ptunet.losses import CompositeReconstructionLoss, LossWeights


def test_composite_loss_is_zero_up_to_charbonnier_floor() -> None:
    values = torch.ones(2, 8, 8)
    objective = CompositeReconstructionLoss()

    output = objective(values, values)

    assert output.components["mse"] == 0.0
    assert output.components["charbonnier"] == pytest.approx(1.0e-3)
    assert output.total.item() == pytest.approx(3.0e-4)


def test_optional_structure_terms_propagate_gradients() -> None:
    prediction = torch.randn(2, 8, 8, requires_grad=True)
    target = torch.randn(2, 8, 8)
    objective = CompositeReconstructionLoss(
        LossWeights(mse=0.5, charbonnier=0.2, gradient=0.2, spectral=0.1)
    )

    output = objective(prediction, target, correction=prediction)
    output.total.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_invalid_loss_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LossWeights(mse=-1.0)
