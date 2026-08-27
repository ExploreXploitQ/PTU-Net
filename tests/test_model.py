from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
import torch

from ptunet.models import (
    OptimizedPatchTransformer,
    PTUNet,
    PTUNetConfig,
    migrate_legacy_state_dict,
    temporal_center_difference,
)


@pytest.fixture(scope="module", autouse=True)
def _limit_torch_threads_for_small_models() -> Iterator[None]:
    """Avoid large OpenMP launch overhead for the tiny CPU test tensors."""

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config(**overrides: object) -> PTUNetConfig:
    config = PTUNetConfig(
        patch_size=8,
        subpatch_size=4,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        baseline_hidden_channels=4,
        unet_base_channels=4,
    )
    return replace(config, **overrides)


def test_forward_shape_and_gradients() -> None:
    torch.manual_seed(7)
    model = PTUNet(small_config())
    inputs = torch.randn(2, 3, 8, 12, requires_grad=True)

    output = model(inputs)

    assert output.shape == (2, 8, 12)
    assert output.dtype == inputs.dtype
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert model.input_projection is not None
    assert model.input_projection.weight.grad is not None
    assert model.correction_scale_logit is not None
    assert model.correction_scale_logit.grad is not None
    assert model.unet_scale is not None
    assert model.unet_scale.grad is not None


def test_raw_frame_mode_matches_explicit_center_difference() -> None:
    torch.manual_seed(11)
    feature_model = PTUNet(small_config(input_mode="center_difference"))
    raw_model = PTUNet(small_config(input_mode="raw_frames"))
    raw_model.load_state_dict(feature_model.state_dict())
    feature_model.eval()
    raw_model.eval()
    frames = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        expected = feature_model(temporal_center_difference(frames))
        actual = raw_model(frames)

    torch.testing.assert_close(actual, expected)


def test_diagnostics_reconstruct_each_residual_stage() -> None:
    torch.manual_seed(13)
    config = small_config()
    model = PTUNet(config).eval()
    inputs = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        prediction, diagnostics = model.forward_with_diagnostics(inputs)

    torch.testing.assert_close(
        diagnostics.pre_refinement,
        diagnostics.baseline + diagnostics.correction_scale * diagnostics.correction_map,
    )
    torch.testing.assert_close(
        prediction,
        diagnostics.pre_refinement
        + diagnostics.unet_scale * diagnostics.unet_residual * diagnostics.unet_gate,
    )
    torch.testing.assert_close(
        diagnostics.spatial_baseline_weights.sum(dim=1),
        torch.ones_like(diagnostics.baseline),
    )
    torch.testing.assert_close(
        diagnostics.global_baseline_weights.sum(),
        torch.ones((), dtype=prediction.dtype),
    )
    assert 0.0 < diagnostics.correction_scale.item() < config.correction_scale_max
    assert torch.all((diagnostics.unet_gate >= 0.0) & (diagnostics.unet_gate <= 1.0))


def test_baseline_only_ablation_has_closed_form_output() -> None:
    config = small_config(
        use_adaptive_baseline=False,
        learnable_baseline_prior=False,
        use_transformer_correction=False,
        use_unet_refinement=False,
    )
    model = PTUNet(config)
    inputs = torch.randn(3, 3, 8, 8)

    prediction, diagnostics = model.forward_with_diagnostics(inputs)

    prior = torch.tensor(config.baseline_prior, dtype=inputs.dtype)
    prior = prior / prior.sum()
    expected = inputs[:, 0] + prior[0] * inputs[:, 1] + prior[2] * inputs[:, 2]
    torch.testing.assert_close(prediction, expected)
    torch.testing.assert_close(diagnostics.baseline, expected)
    assert diagnostics.correction_scale.item() == 0.0
    assert diagnostics.unet_scale.item() == 0.0
    assert torch.count_nonzero(diagnostics.correction_map) == 0
    assert torch.count_nonzero(diagnostics.unet_residual) == 0


def test_dynamic_odd_resolution_uses_interpolated_positions() -> None:
    model = PTUNet(
        small_config(
            patch_size=6,
            subpatch_size=3,
            embed_dim=12,
            num_heads=3,
            num_layers=1,
        )
    ).eval()
    inputs = torch.randn(1, 3, 9, 15)

    with torch.no_grad():
        output = model(inputs)

    assert output.shape == (1, 9, 15)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"patch_size": 15, "subpatch_size": 4}, "divisible"),
        ({"embed_dim": 30, "num_heads": 8}, "num_heads"),
        ({"correction_scale_init": 0.2, "correction_scale_max": 0.2}, "strictly"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_config_rejects_invalid_architectures(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        small_config(**overrides)


def test_forward_rejects_invalid_input_contract() -> None:
    model = PTUNet(small_config())

    with pytest.raises(ValueError, match="exactly 3"):
        model(torch.randn(1, 2, 8, 8))
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 3, 9, 8))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.ones(1, 3, 8, 8, dtype=torch.int64))


def _legacy_key(key: str) -> str:
    if key == "global_baseline_logits":
        return "compressor_weights"
    if key == "baseline_mix_logit":
        return "baseline_mix"
    if key == "position_embedding":
        return "pos_embedding"
    if key.startswith("transformer_blocks."):
        return "blocks." + key.removeprefix("transformer_blocks.")
    replacements = (
        ("unet_head.encoder1.", "unet_head.enc1."),
        ("unet_head.encoder2.", "unet_head.enc2."),
        ("unet_head.upsample.", "unet_head.up1."),
        ("unet_head.decoder.", "unet_head.dec1."),
        ("unet_head.output.", "unet_head.out."),
    )
    for current_prefix, legacy_prefix in replacements:
        if key.startswith(current_prefix):
            key = legacy_prefix + key.removeprefix(current_prefix)
            break
    if key.startswith("unet_head."):
        key = key.replace(".projection.", ".fc.")
    return key


def test_legacy_format_v2_state_dict_migration() -> None:
    torch.manual_seed(17)
    config = small_config()
    source = PTUNet(config).eval()
    legacy_state = {}
    for key, value in source.state_dict().items():
        legacy_key = _legacy_key(key)
        legacy_state[legacy_key] = (
            torch.sigmoid(value) if key == "baseline_mix_logit" else value.clone()
        )

    migrated = migrate_legacy_state_dict(
        {
            "model_state_dict": legacy_state,
            "normalization_params": {"data_mean": 0.0, "data_std": 1.0},
            "format_version": 2,
        }
    )
    assert set(migrated) == set(source.state_dict())

    restored = PTUNet(config).eval()
    report = restored.load_legacy_state_dict({"model_state_dict": legacy_state})
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    inputs = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), source(inputs))

    assert OptimizedPatchTransformer is PTUNet
