from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from ptunet.checkpoint import CheckpointMetadata, save_checkpoint
from ptunet.config import (
    DatasetSpec,
    ExperimentConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
)
from ptunet.engine import EpochRecord
from ptunet.experiment import (
    _normalization_from_checkpoint,
    _validate_checkpoint_model_config,
    _validate_resume_config,
    train_experiment,
)
from ptunet.factory import build_model
from ptunet.io import CompressorResolver, compressed_path, original_path, write_float32_atomic


def _config(tmp_path) -> ExperimentConfig:
    spec = DatasetSpec(
        name="field",
        variable="v",
        resolution_tag="4x4",
        height=4,
        width=4,
        train_timesteps=(0,),
        validation_timesteps=(1,),
        test_timesteps=(2,),
        even_compressor="sz3",
        odd_compressor="sz3",
    )
    return ExperimentConfig(
        paths=PathConfig(tmp_path / "data", tmp_path / "runs"),
        datasets=(spec,),
        compressor_extensions={"sz3": ".sz3.out"},
        model=ModelConfig(
            patch_size=4,
            stride=2,
            subpatch_size=2,
            embed_dim=8,
            num_heads=2,
            num_layers=1,
            unet_base_channels=4,
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=1,
            validation_batch_size=1,
            num_workers=0,
            baseline_warmup_epochs=0,
            mixed_precision=False,
            device="cpu",
        ),
    )


def _metadata(config: ExperimentConfig) -> CheckpointMetadata:
    return CheckpointMetadata(
        format_version=2,
        epoch=0,
        global_step=1,
        best_metric=0.1,
        best_epoch=0,
        patience_counter=0,
        stopped_early=False,
        is_best_checkpoint=False,
        normalization=None,
        experiment=config.to_dict(),
        history=(),
        best_model_state_dict=None,
        data_loader_state=None,
    )


def test_checkpoint_model_mismatch_is_rejected_but_runtime_data_overrides_are_safe(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    metadata = _metadata(config)
    different_stride = replace(config, model=replace(config.model, stride=1))

    with pytest.raises(ValueError, match=r"model\.stride"):
        _validate_checkpoint_model_config(metadata, different_stride)

    relocated = replace(
        config,
        paths=PathConfig(tmp_path / "relocated-data", tmp_path / "different-output"),
    )
    _validate_checkpoint_model_config(metadata, relocated)

    with pytest.raises(ValueError, match="does not record model configuration"):
        _validate_checkpoint_model_config(replace(metadata, experiment={}), config)


def test_resume_allows_runtime_controls_and_rejects_stateful_training_changes(tmp_path) -> None:
    config = _config(tmp_path)
    metadata = _metadata(config)
    runtime_override = replace(
        config,
        training=replace(
            config.training,
            epochs=2,
            batch_size=3,
            validation_batch_size=2,
            num_workers=1,
            device="auto",
        ),
    )
    _validate_resume_config(metadata, runtime_override)

    changed_optimizer = replace(
        config,
        training=replace(config.training, learning_rate=2.0e-4),
    )
    with pytest.raises(ValueError, match=r"training\.learning_rate"):
        _validate_resume_config(metadata, changed_optimizer)


def test_malformed_checkpoint_normalization_never_silently_refits(tmp_path) -> None:
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="normalization metadata is malformed"):
        _normalization_from_checkpoint({"mean": 0.0, "std": 1.0}, config)
    with pytest.raises(ValueError, match="does not contain normalization"):
        _normalization_from_checkpoint(None, config)


def test_train_resume_uses_checkpoint_normalization_without_test_files(tmp_path) -> None:
    config = _config(tmp_path)
    spec = config.datasets[0]
    resolver = CompressorResolver(config.compressor_extensions)
    for timestep in (0, 1):
        compressed = np.full(spec.shape, timestep + 0.5, dtype=np.float32)
        target = np.full(spec.shape, timestep + 1.0, dtype=np.float32)
        write_float32_atomic(
            compressed_path(spec, config.paths.data_root, timestep, resolver),
            compressed,
        )
        write_float32_atomic(original_path(spec, config.paths.data_root, timestep), target)

    model = build_model(config.model)
    history = EpochRecord(
        epoch=0,
        train={"loss": 0.2},
        validation={"loss": 0.1},
        learning_rates=[config.training.learning_rate],
        duration_seconds=1.0,
        improved=True,
    )
    checkpoint = tmp_path / "prior" / "last.pt"
    normalization = {
        "mean": 10.0,
        "std": 2.0,
        "count": 16,
        "minimum": 0.0,
        "maximum": 20.0,
    }
    save_checkpoint(
        checkpoint,
        model,
        epoch=0,
        global_step=1,
        best_metric=0.1,
        best_epoch=0,
        normalization=normalization,
        experiment=config.to_dict(),
        history=[history.__dict__],
        best_model_state_dict=model.state_dict(),
    )
    output = tmp_path / "continued"

    run = train_experiment(config, output_directory=output, resume_checkpoint=checkpoint)

    assert run.datasets.normalization.mean == 10.0
    assert run.datasets.normalization.std == 2.0
    assert run.datasets.test is None
    assert run.result.epochs_completed == 1
    assert run.result.best_epoch == 0
    assert run.best_checkpoint.is_file()
    assert run.last_checkpoint.is_file()
    assert len((output / "history.csv").read_text(encoding="utf-8").splitlines()) == 2
    assert len((output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["epochs_completed"] == 1
    resume = json.loads((output / "resume.json").read_text(encoding="utf-8"))
    assert resume["normalization_source"] == "checkpoint"
