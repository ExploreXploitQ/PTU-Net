from __future__ import annotations

import pytest
import torch

from ptunet.models import PTUNet, PTUNetConfig


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_ptunet_forward_backward_on_cuda() -> None:
    device = torch.device("cuda:0")
    model = PTUNet(
        PTUNetConfig(
            patch_size=16,
            subpatch_size=4,
            embed_dim=32,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            baseline_hidden_channels=8,
            unet_base_channels=8,
        )
    ).to(device)
    inputs = torch.randn(2, 3, 16, 16, device=device)

    prediction, diagnostics = model.forward_with_diagnostics(inputs)
    loss = prediction.square().mean() + 1.0e-4 * diagnostics.correction_map.square().mean()
    loss.backward()

    assert prediction.shape == (2, 16, 16)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
