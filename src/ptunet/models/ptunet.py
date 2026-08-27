"""Core Patch Transformer U-Net model.

The implementation in this module is deliberately independent of datasets,
training loops, filesystem layout, and experiment tracking.  Inputs are
normalized temporal patches with shape ``[batch, 3, height, width]``.  By
default, their channels are ``[center, previous - center, next - center]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

InputMode = Literal["center_difference", "raw_frames"]


def temporal_center_difference(frames: Tensor) -> Tensor:
    """Convert ``[previous, center, next]`` frames to center-difference features.

    Args:
        frames: Floating-point tensor with shape ``[B, 3, H, W]``.

    Returns:
        Tensor with channels ``[center, previous - center, next - center]``.
    """

    if frames.ndim != 4:
        raise ValueError(
            f"temporal input must have shape [B, 3, H, W]; received rank {frames.ndim}"
        )
    if frames.shape[1] != 3:
        raise ValueError(
            "temporal input must contain previous, center, and next frames; "
            f"received {frames.shape[1]} channels"
        )
    if not torch.is_floating_point(frames):
        raise TypeError("temporal input must be a floating-point tensor")

    previous = frames[:, 0:1]
    center = frames[:, 1:2]
    following = frames[:, 2:3]
    return torch.cat((center, previous - center, following - center), dim=1)


@dataclass(frozen=True)
class PTUNetConfig:
    """Configuration for :class:`PTUNet`.

    Defaults reproduce the model dimensions and initialization policy from the
    original research script.  The three ``use_*`` pathway flags make baseline,
    transformer-correction, and U-Net ablations explicit and checkpointable.
    ``patch_size`` defines the learned positional-embedding grid; inference may
    use any spatial size divisible by ``subpatch_size``.
    """

    patch_size: int = 32
    subpatch_size: int = 8
    embed_dim: int = 384
    num_heads: int = 12
    num_layers: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    input_mode: InputMode = "center_difference"

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
        positive_integers = {
            "patch_size": self.patch_size,
            "subpatch_size": self.subpatch_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "baseline_hidden_channels": self.baseline_hidden_channels,
            "unet_base_channels": self.unet_base_channels,
            "squeeze_excite_reduction": self.squeeze_excite_reduction,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {self.num_layers}")
        if self.patch_size % self.subpatch_size != 0:
            raise ValueError(
                "patch_size must be divisible by subpatch_size: "
                f"{self.patch_size} % {self.subpatch_size} != 0"
            )
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads: "
                f"{self.embed_dim} % {self.num_heads} != 0"
            )
        if self.use_transformer_correction and self.embed_dim < 2:
            raise ValueError("embed_dim must be at least two when correction is enabled")
        if self.mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.input_mode not in ("center_difference", "raw_frames"):
            raise ValueError(
                f"input_mode must be 'center_difference' or 'raw_frames', got {self.input_mode!r}"
            )
        if len(self.baseline_prior) != 3 or any(value <= 0.0 for value in self.baseline_prior):
            raise ValueError("baseline_prior must contain three strictly positive values")
        if sum(self.baseline_prior) <= 0.0:
            raise ValueError("baseline_prior must have positive total mass")
        if not 0.0 < self.baseline_mix_init < 1.0:
            raise ValueError("baseline_mix_init must be strictly between zero and one")
        if self.correction_scale_max <= 0.0:
            raise ValueError("correction_scale_max must be positive")
        if not 0.0 < self.correction_scale_init < self.correction_scale_max:
            raise ValueError(
                "correction_scale_init must be strictly between zero and correction_scale_max"
            )
        if self.linear_init_gain <= 0.0:
            raise ValueError("linear_init_gain must be positive")


@dataclass(frozen=True)
class PTUNetDiagnostics:
    """Intermediate tensors returned by :meth:`PTUNet.forward_with_diagnostics`.

    Tensors remain attached to the autograd graph so callers can construct
    auxiliary losses, such as correction-map regularization, without relying
    on mutable ``last_*`` attributes.
    """

    baseline: Tensor
    spatial_baseline_weights: Tensor
    global_baseline_weights: Tensor
    baseline_mix: Tensor
    correction_map: Tensor
    correction_scale: Tensor
    pre_refinement: Tensor
    unet_residual: Tensor
    unet_gate: Tensor
    unet_scale: Tensor


@dataclass(frozen=True)
class StateDictLoadReport:
    """Public, stable summary of a legacy state-dict load."""

    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def migrate_legacy_state_dict(checkpoint: Mapping[str, object]) -> dict[str, Tensor]:
    """Translate a format-version-2 prototype checkpoint to current key names.

    ``checkpoint`` may be either the raw model state dictionary or the wrapper
    written by the original script, which contains ``model_state_dict``,
    ``normalization_params``, and ``format_version``.  Normalization metadata is
    intentionally left to the data/inference layer.
    """

    wrapped_state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(wrapped_state, Mapping):
        raise TypeError("legacy checkpoint does not contain a model state dictionary")

    migrated: dict[str, Tensor] = {}
    for raw_key, raw_value in wrapped_state.items():
        if not isinstance(raw_key, str):
            raise TypeError("legacy state-dict keys must be strings")
        if not isinstance(raw_value, Tensor):
            raise TypeError(f"legacy state-dict value for {raw_key!r} is not a tensor")

        key = raw_key.removeprefix("module.")
        value = raw_value
        if key == "compressor_weights":
            key = "global_baseline_logits"
        elif key == "baseline_mix":
            key = "baseline_mix_logit"
            epsilon = torch.finfo(value.dtype).eps
            value = torch.logit(value.clamp(min=epsilon, max=1.0 - epsilon))
        elif key == "pos_embedding":
            key = "position_embedding"

        if key.startswith("blocks."):
            key = "transformer_blocks." + key.removeprefix("blocks.")
        if key.startswith("unet_head."):
            replacements = (
                ("unet_head.enc1.", "unet_head.encoder1."),
                ("unet_head.enc2.", "unet_head.encoder2."),
                ("unet_head.up1.", "unet_head.upsample."),
                ("unet_head.dec1.", "unet_head.decoder."),
                ("unet_head.out.", "unet_head.output."),
            )
            for old_prefix, new_prefix in replacements:
                if key.startswith(old_prefix):
                    key = new_prefix + key.removeprefix(old_prefix)
                    break
            key = key.replace(".fc.", ".projection.")

        if key in migrated:
            raise ValueError(f"legacy keys collide after migration at {key!r}")
        migrated[key] = value
    return migrated


def _group_count(channels: int, maximum: int = 8) -> int:
    """Choose a useful GroupNorm group count that divides ``channels``."""

    upper = min(maximum, max(1, channels // 4))
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.projection(self.pool(inputs))


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = _group_count(out_channels)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class _AttentionGatedUNet(nn.Module):
    """One-level U-Net used to predict a gated residual and output gate."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        *,
        use_skip_attention: bool,
        use_squeeze_excite: bool,
        squeeze_excite_reduction: int,
    ) -> None:
        super().__init__()
        self.use_skip_attention = use_skip_attention

        first_attention: nn.Module
        second_attention: nn.Module
        decoder_attention: nn.Module
        if use_squeeze_excite:
            first_attention = _SqueezeExcite(base_channels, squeeze_excite_reduction)
            second_attention = _SqueezeExcite(base_channels * 2, squeeze_excite_reduction)
            decoder_attention = _SqueezeExcite(base_channels, squeeze_excite_reduction)
        else:
            first_attention = nn.Identity()
            second_attention = nn.Identity()
            decoder_attention = nn.Identity()

        self.encoder1 = nn.Sequential(
            _ConvBlock(in_channels, base_channels),
            first_attention,
        )
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = nn.Sequential(
            _ConvBlock(base_channels, base_channels * 2),
            second_attention,
        )
        self.upsample = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )
        self.skip_gate = nn.Conv2d(base_channels * 2, 1, kernel_size=1)
        self.decoder = nn.Sequential(
            _ConvBlock(base_channels * 2, base_channels),
            decoder_attention,
        )
        self.output = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.encoder1(inputs)
        encoded = self.encoder2(self.downsample(skip))
        upsampled = self.upsample(encoded)
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.use_skip_attention:
            attention = torch.sigmoid(self.skip_gate(torch.cat((upsampled, skip), dim=1)))
            skip = skip * attention

        decoded = self.decoder(torch.cat((upsampled, skip), dim=1))
        residual, gate_logits = torch.chunk(self.output(decoded), chunks=2, dim=1)
        return residual.squeeze(1), torch.sigmoid(gate_logits.squeeze(1))


class _TransformerBlock(nn.Module):
    """Pre-normalized self-attention block matching the research prototype."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, round(embed_dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class PTUNet(nn.Module):
    """Temporal Patch Transformer U-Net reconstruction model.

    The prediction is assembled in three interpretable stages:

    1. an adaptive center-anchored temporal baseline;
    2. a subpatch-transformer correction with a learned bounded scale; and
    3. an attention-gated U-Net residual refinement.
    """

    input_channels: int = 3

    def __init__(self, config: PTUNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or PTUNetConfig()

        prior = torch.tensor(self.config.baseline_prior, dtype=torch.float32)
        prior = prior / prior.sum()
        self.global_baseline_logits = nn.Parameter(
            torch.log(prior),
            requires_grad=self.config.learnable_baseline_prior,
        )
        mix = torch.tensor(self.config.baseline_mix_init, dtype=torch.float32)
        self.baseline_mix_logit = nn.Parameter(torch.logit(mix))

        self.baseline_gate: nn.Module | None
        if self.config.use_adaptive_baseline:
            self.baseline_gate = nn.Sequential(
                nn.Conv2d(
                    self.input_channels,
                    self.config.baseline_hidden_channels,
                    kernel_size=1,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.config.baseline_hidden_channels,
                    self.input_channels,
                    kernel_size=1,
                ),
            )
        else:
            self.baseline_gate = None

        self.input_projection: nn.Linear | None
        self.position_embedding: nn.Parameter | None
        self.transformer_blocks: nn.ModuleList
        self.correction_head: nn.Sequential | None
        self.correction_scale_logit: nn.Parameter | None
        if self.config.use_transformer_correction:
            token_features = self.input_channels * self.config.subpatch_size**2
            self.input_projection = nn.Linear(token_features, self.config.embed_dim)
            base_grid = self.config.patch_size // self.config.subpatch_size
            if self.config.use_positional_embedding:
                self.position_embedding = nn.Parameter(
                    torch.randn(1, base_grid * base_grid, self.config.embed_dim) * 0.02
                )
            else:
                self.register_parameter("position_embedding", None)
            self.transformer_blocks = nn.ModuleList(
                [
                    _TransformerBlock(
                        self.config.embed_dim,
                        self.config.num_heads,
                        self.config.mlp_ratio,
                        self.config.dropout,
                    )
                    for _ in range(self.config.num_layers)
                ]
            )
            self.correction_head = nn.Sequential(
                nn.Linear(self.config.embed_dim, self.config.embed_dim // 2),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(
                    self.config.embed_dim // 2,
                    self.config.subpatch_size**2,
                ),
            )
            scale_ratio = torch.tensor(
                self.config.correction_scale_init / self.config.correction_scale_max,
                dtype=torch.float32,
            )
            self.correction_scale_logit = nn.Parameter(torch.logit(scale_ratio))
        else:
            self.input_projection = None
            self.register_parameter("position_embedding", None)
            self.transformer_blocks = nn.ModuleList()
            self.correction_head = None
            self.register_parameter("correction_scale_logit", None)

        self.unet_head: _AttentionGatedUNet | None
        self.unet_scale: nn.Parameter | None
        if self.config.use_unet_refinement:
            self.unet_head = _AttentionGatedUNet(
                in_channels=4,
                base_channels=self.config.unet_base_channels,
                use_skip_attention=self.config.use_skip_attention,
                use_squeeze_excite=self.config.use_squeeze_excite,
                squeeze_excite_reduction=self.config.squeeze_excite_reduction,
            )
            self.unet_scale = nn.Parameter(
                torch.tensor(self.config.unet_scale_init, dtype=torch.float32)
            )
        else:
            self.unet_head = None
            self.register_parameter("unet_scale", None)

        self.apply(self._initialize_linear)
        self._initialize_baseline_gate_prior()

    def _initialize_linear(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=self.config.linear_init_gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _initialize_baseline_gate_prior(self) -> None:
        gate = self.baseline_gate
        if not isinstance(gate, nn.Sequential):
            return
        final_projection = gate[-1]
        if not isinstance(final_projection, nn.Conv2d) or final_projection.bias is None:
            return
        prior = torch.tensor(
            self.config.baseline_prior,
            dtype=final_projection.bias.dtype,
            device=final_projection.bias.device,
        )
        prior = prior / prior.sum()
        with torch.no_grad():
            final_projection.weight.zero_()
            final_projection.bias.copy_(torch.log(prior))

    def _validate_input(self, inputs: Tensor) -> None:
        if inputs.ndim != 4:
            raise ValueError(
                f"PTUNet input must have shape [B, 3, H, W]; received rank {inputs.ndim}"
            )
        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"PTUNet requires exactly {self.input_channels} temporal channels, "
                f"received {inputs.shape[1]}"
            )
        if not torch.is_floating_point(inputs):
            raise TypeError("PTUNet input must be a floating-point tensor")
        height, width = inputs.shape[-2:]
        subpatch = self.config.subpatch_size
        if height < subpatch or width < subpatch:
            raise ValueError(
                f"input spatial size {(height, width)} is smaller than subpatch_size {subpatch}"
            )
        if height % subpatch != 0 or width % subpatch != 0:
            raise ValueError(
                "input height and width must be divisible by subpatch_size: "
                f"received {(height, width)} and {subpatch}"
            )
        if self.config.use_unet_refinement and (height < 2 or width < 2):
            raise ValueError("U-Net refinement requires height and width of at least two")

    def _model_features(self, inputs: Tensor) -> Tensor:
        if self.config.input_mode == "raw_frames":
            return temporal_center_difference(inputs)
        return inputs

    def current_global_baseline_weights(self) -> Tensor:
        """Return the normalized learned global temporal weights."""

        return F.softmax(self.global_baseline_logits, dim=0)

    def current_baseline_mix(self) -> Tensor:
        """Return the bounded adaptive/global baseline mixing coefficient."""

        if not self.config.use_adaptive_baseline:
            return self.global_baseline_logits.new_zeros(())
        return torch.sigmoid(self.baseline_mix_logit)

    def current_correction_scale(self) -> Tensor:
        """Return the learned correction scale in ``(0, correction_scale_max)``."""

        scale_logit = self.correction_scale_logit
        if scale_logit is None:
            return self.global_baseline_logits.new_zeros(())
        return self.config.correction_scale_max * torch.sigmoid(scale_logit)

    def load_legacy_state_dict(
        self,
        checkpoint: Mapping[str, object],
        *,
        strict: bool = True,
    ) -> StateDictLoadReport:
        """Load a raw or wrapped format-version-2 prototype state dictionary."""

        result = self.load_state_dict(
            migrate_legacy_state_dict(checkpoint),
            strict=strict,
        )
        return StateDictLoadReport(
            missing_keys=tuple(result.missing_keys),
            unexpected_keys=tuple(result.unexpected_keys),
        )

    def _adaptive_baseline(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        global_weights = self.current_global_baseline_weights()
        broadcast_global = global_weights.view(1, -1, 1, 1)
        mix = self.current_baseline_mix()
        gate = self.baseline_gate
        if gate is None:
            weights = broadcast_global.expand_as(features)
        else:
            spatial = F.softmax(gate(features), dim=1)
            weights = mix * spatial + (1.0 - mix) * broadcast_global

        center = features[:, 0]
        previous_delta = features[:, 1]
        next_delta = features[:, 2]
        baseline = center + weights[:, 0] * previous_delta + weights[:, 2] * next_delta
        return baseline, weights, global_weights, mix

    def _positional_tokens(self, rows: int, columns: int, reference: Tensor) -> Tensor:
        position_embedding = self.position_embedding
        if position_embedding is None:
            return reference.new_zeros((1, rows * columns, self.config.embed_dim))

        base_grid = self.config.patch_size // self.config.subpatch_size
        position_grid = position_embedding.reshape(
            1,
            base_grid,
            base_grid,
            self.config.embed_dim,
        ).permute(0, 3, 1, 2)
        if (rows, columns) != (base_grid, base_grid):
            position_grid = F.interpolate(
                position_grid,
                size=(rows, columns),
                mode="bicubic",
                align_corners=False,
            )
        return (
            position_grid.permute(0, 2, 3, 1)
            .reshape(1, rows * columns, self.config.embed_dim)
            .to(dtype=reference.dtype)
        )

    def _transformer_correction(self, features: Tensor) -> tuple[Tensor, Tensor]:
        input_projection = self.input_projection
        correction_head = self.correction_head
        if input_projection is None or correction_head is None:
            return torch.zeros_like(features[:, 0]), self.current_correction_scale()

        subpatch = self.config.subpatch_size
        height, width = features.shape[-2:]
        rows, columns = height // subpatch, width // subpatch
        patches = F.unfold(
            features,
            kernel_size=subpatch,
            stride=subpatch,
        ).transpose(1, 2)
        tokens = input_projection(patches)
        tokens = tokens + self._positional_tokens(rows, columns, tokens)
        for block in self.transformer_blocks:
            tokens = block(tokens)

        correction_patches = correction_head(tokens).transpose(1, 2)
        correction_map = F.fold(
            correction_patches,
            output_size=(height, width),
            kernel_size=subpatch,
            stride=subpatch,
        ).squeeze(1)
        return correction_map, self.current_correction_scale()

    def forward_with_diagnostics(self, inputs: Tensor) -> tuple[Tensor, PTUNetDiagnostics]:
        """Run the model and return interpretable intermediate tensors."""

        self._validate_input(inputs)
        features = self._model_features(inputs)
        baseline, spatial_weights, global_weights, baseline_mix = self._adaptive_baseline(features)
        correction_map, correction_scale = self._transformer_correction(features)
        pre_refinement = baseline + correction_scale * correction_map

        unet_head = self.unet_head
        unet_scale_parameter = self.unet_scale
        if unet_head is None or unet_scale_parameter is None:
            unet_residual = torch.zeros_like(pre_refinement)
            unet_gate = torch.zeros_like(pre_refinement)
            unet_scale = pre_refinement.new_zeros(())
            prediction = pre_refinement
        else:
            unet_inputs = torch.stack(
                (
                    pre_refinement,
                    baseline,
                    features[:, 1],
                    features[:, 2],
                ),
                dim=1,
            )
            unet_residual, unet_gate = unet_head(unet_inputs)
            unet_scale = unet_scale_parameter
            prediction = pre_refinement + unet_scale * unet_residual * unet_gate

        diagnostics = PTUNetDiagnostics(
            baseline=baseline,
            spatial_baseline_weights=spatial_weights,
            global_baseline_weights=global_weights,
            baseline_mix=baseline_mix,
            correction_map=correction_map,
            correction_scale=correction_scale,
            pre_refinement=pre_refinement,
            unet_residual=unet_residual,
            unet_gate=unet_gate,
            unet_scale=unet_scale,
        )
        return prediction, diagnostics

    def forward(self, inputs: Tensor) -> Tensor:
        """Reconstruct a normalized center frame as a tensor of shape ``[B, H, W]``."""

        prediction, _ = self.forward_with_diagnostics(inputs)
        return prediction


# Import-compatible name used by the original monolithic training script.
OptimizedPatchTransformer = PTUNet


__all__ = [
    "InputMode",
    "OptimizedPatchTransformer",
    "PTUNet",
    "PTUNetConfig",
    "PTUNetDiagnostics",
    "StateDictLoadReport",
    "migrate_legacy_state_dict",
    "temporal_center_difference",
]
