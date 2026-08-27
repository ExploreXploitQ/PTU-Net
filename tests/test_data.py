from __future__ import annotations

import numpy as np
import pytest
import torch

from ptunet.config import (
    DatasetSpec,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
)
from ptunet.data import (
    NormalizationStats,
    UnifiedPatchDataset,
    build_dataset_splits,
    center_difference_features,
    patch_coverage_counts,
    resolve_temporal_window,
)
from ptunet.io import (
    CompressorResolver,
    FieldIOError,
    compressed_path,
    open_float32_memmap,
    original_path,
    write_float32_atomic,
)


def _spec() -> DatasetSpec:
    return DatasetSpec(
        name="field",
        variable="v",
        resolution_tag="4x5",
        height=4,
        width=5,
        train_timesteps=(1, 2),
        validation_timesteps=(3,),
        test_timesteps=(4,),
        even_compressor="sz3",
        odd_compressor="hpez",
    )


def _materialize_fields(tmp_path, spec, resolver) -> None:
    for timestep in range(6):
        compressed = np.full(spec.shape, timestep + 0.25, dtype=np.float32)
        write_float32_atomic(compressed_path(spec, tmp_path, timestep, resolver), compressed)
    targets = {
        1: np.full(spec.shape, 1.0, dtype=np.float32),
        2: np.full(spec.shape, 2.0, dtype=np.float32),
        3: np.full(spec.shape, 1_000.0, dtype=np.float32),
        4: np.full(spec.shape, -1_000.0, dtype=np.float32),
    }
    for timestep, target in targets.items():
        write_float32_atomic(original_path(spec, tmp_path, timestep), target)


def test_float32_memmap_is_lazy_and_exact_size(tmp_path) -> None:
    path = tmp_path / "field.dat"
    write_float32_atomic(path, np.arange(12, dtype=np.float32).reshape(3, 4))

    field = open_float32_memmap(path, (3, 4))

    assert isinstance(field, np.memmap)
    assert field.dtype == np.dtype("<f4")
    assert field[2, 3] == 11
    with pytest.raises(FieldIOError, match="expected exactly"):
        open_float32_memmap(path, (4, 4))


def test_compressor_resolution_uses_timestep_parity(tmp_path) -> None:
    spec = _spec()
    resolver = CompressorResolver({"sz3": ".sz3.out", "hpez": ".hpez.out"})

    even = compressed_path(spec, tmp_path, 2, resolver)
    odd = compressed_path(spec, tmp_path, 3, resolver)

    assert even.name == "v_4x5_0002.dat.sz3.out"
    assert odd.name == "v_4x5_0003.dat.hpez.out"


def test_temporal_boundaries_preserve_center_feature_channel_zero() -> None:
    assert resolve_temporal_window(0, [0, 1, 2]).timesteps == (2, 0, 1)
    assert resolve_temporal_window(2, [0, 1, 2]).timesteps == (1, 2, 0)
    assert resolve_temporal_window(0, [0, 1]).timesteps == (0, 0, 1)
    stats = NormalizationStats(mean=10.0, std=2.0, count=3, minimum=8.0, maximum=12.0)
    previous = np.full((2, 2), 12.0, dtype=np.float32)
    center = np.full((2, 2), 10.0, dtype=np.float32)
    following = np.full((2, 2), 8.0, dtype=np.float32)

    features = center_difference_features(previous, center, following, stats)

    np.testing.assert_array_equal(features[0], np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(features[1], np.ones((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(features[2], -np.ones((2, 2), dtype=np.float32))


def test_patch_positions_cover_irregular_edges() -> None:
    coverage = patch_coverage_counts((7, 10), patch_size=4, stride=3)

    assert coverage.shape == (7, 10)
    assert np.all(coverage > 0)
    with pytest.raises(ValueError, match="exceeds"):
        patch_coverage_counts((3, 10), patch_size=4, stride=2)
    with pytest.raises(ValueError, match="leave gaps"):
        patch_coverage_counts((10, 10), patch_size=3, stride=4)


def test_build_splits_reuses_train_statistics_and_memmaps_on_demand(tmp_path) -> None:
    spec = _spec()
    extensions = {"sz3": ".sz3.out", "hpez": ".hpez.out"}
    resolver = CompressorResolver(extensions)
    _materialize_fields(tmp_path, spec, resolver)
    config = ExperimentConfig(
        paths=PathConfig(tmp_path, tmp_path / "output"),
        datasets=(spec,),
        compressor_extensions=extensions,
        model=ModelConfig(patch_size=4, stride=3, subpatch_size=2),
        training=TrainingConfig(normalization_chunk_elements=7),
        evaluation=EvaluationConfig(temporal_search_radius=2),
    )

    bundle = build_dataset_splits(config)

    assert bundle.normalization.mean == pytest.approx(1.5)
    assert bundle.normalization.std == pytest.approx(0.5)
    assert bundle.normalization.count == 40
    assert bundle.validation is not None
    assert bundle.test is not None
    assert bundle.validation.normalization is bundle.normalization
    assert bundle.test.normalization is bundle.normalization
    assert bundle.train._mmap_cache == {}

    inputs, target = bundle.train[0]

    assert inputs.shape == (3, 4, 4)
    assert target.shape == (4, 4)
    assert inputs.dtype == torch.float32
    assert target.dtype == torch.float32
    assert bundle.train._mmap_cache
    expected_center = (1.25 - 1.5) / 0.5
    torch.testing.assert_close(inputs[0], torch.full((4, 4), expected_center))


def test_validation_requires_explicit_training_statistics(tmp_path) -> None:
    spec = _spec()
    extensions = {"sz3": ".sz3.out", "hpez": ".hpez.out"}
    resolver = CompressorResolver(extensions)
    _materialize_fields(tmp_path, spec, resolver)

    with pytest.raises(ValueError, match="require training normalization"):
        UnifiedPatchDataset(
            (spec,),
            tmp_path,
            resolver,
            split="validation",
            patch_size=4,
            stride=2,
        )


def test_raw_frame_mode_emits_normalized_previous_center_next(tmp_path) -> None:
    spec = _spec()
    extensions = {"sz3": ".sz3.out", "hpez": ".hpez.out"}
    resolver = CompressorResolver(extensions)
    _materialize_fields(tmp_path, spec, resolver)
    stats = NormalizationStats(mean=1.5, std=0.5, count=40, minimum=1.0, maximum=2.0)
    dataset = UnifiedPatchDataset(
        (spec,),
        tmp_path,
        resolver,
        split="validation",
        patch_size=4,
        stride=2,
        normalization=stats,
        input_mode="raw_frames",
    )

    inputs, _ = dataset[0]

    torch.testing.assert_close(inputs[0], torch.full((4, 4), 1.5))
    torch.testing.assert_close(inputs[1], torch.full((4, 4), 3.5))
    torch.testing.assert_close(inputs[2], torch.full((4, 4), 5.5))


def test_empty_optional_splits_are_explicitly_none(tmp_path) -> None:
    base = _spec()
    spec = DatasetSpec(
        name=base.name,
        variable=base.variable,
        resolution_tag=base.resolution_tag,
        height=base.height,
        width=base.width,
        train_timesteps=base.train_timesteps,
        even_compressor=base.even_compressor,
        odd_compressor=base.odd_compressor,
    )
    extensions = {"sz3": ".sz3.out", "hpez": ".hpez.out"}
    resolver = CompressorResolver(extensions)
    _materialize_fields(tmp_path, base, resolver)
    config = ExperimentConfig(
        paths=PathConfig(tmp_path, tmp_path / "output"),
        datasets=(spec,),
        compressor_extensions=extensions,
        model=ModelConfig(patch_size=4, stride=2, subpatch_size=2),
    )

    bundle = build_dataset_splits(config)

    assert bundle.validation is None
    assert bundle.test is None


def test_checkpoint_normalization_is_used_and_training_can_skip_test_files(tmp_path) -> None:
    spec = _spec()
    extensions = {"sz3": ".sz3.out", "hpez": ".hpez.out"}
    resolver = CompressorResolver(extensions)
    _materialize_fields(tmp_path, spec, resolver)
    original_path(spec, tmp_path, 4).unlink()
    stats = NormalizationStats(mean=10.0, std=2.0, count=40, minimum=0.0, maximum=20.0)
    config = ExperimentConfig(
        paths=PathConfig(tmp_path, tmp_path / "output"),
        datasets=(spec,),
        compressor_extensions=extensions,
        model=ModelConfig(patch_size=4, stride=2, subpatch_size=2),
    )

    bundle = build_dataset_splits(config, normalization=stats, include_test=False)

    assert bundle.normalization is stats
    assert bundle.train.normalization is stats
    assert bundle.validation is not None
    assert bundle.validation.normalization is stats
    assert bundle.test is None
    _, target = bundle.train[0]
    torch.testing.assert_close(target, torch.full((4, 4), -4.5))
