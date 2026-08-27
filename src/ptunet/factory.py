"""Factories that connect typed configuration to runtime objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from ptunet.config import ExperimentConfig, ModelConfig, TrainingConfig
from ptunet.data import (
    DatasetBundle,
    NormalizationStats,
    UnifiedPatchDataset,
    build_dataset_splits,
)
from ptunet.engine import TrainerSettings
from ptunet.losses import LossWeights
from ptunet.models import PTUNet, PTUNetConfig
from ptunet.reproducibility import seed_worker


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader[Any]
    validation: DataLoader[Any] | None
    test: DataLoader[Any] | None


def build_model(config: ModelConfig) -> PTUNet:
    """Build PTU-Net from the YAML-facing model configuration."""

    values = asdict(config)
    values.pop("stride")
    return PTUNet(PTUNetConfig(**values))


def build_trainer_settings(config: TrainingConfig) -> TrainerSettings:
    """Translate YAML training fields into the engine's runtime settings."""

    return TrainerSettings(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        scheduler_factor=config.scheduler_factor,
        scheduler_patience=config.scheduler_patience,
        minimum_learning_rate=config.minimum_learning_rate,
        early_stopping_patience=config.early_stopping_patience,
        minimum_delta=config.minimum_delta,
        gradient_clip_norm=config.gradient_clip_norm,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        baseline_warmup_epochs=config.baseline_warmup_epochs,
        mixed_precision=config.mixed_precision,
        device=config.device,
        loss=LossWeights(**asdict(config.loss)),
    )


def _loader(
    dataset: UnifiedPatchDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_loaders(
    config: ExperimentConfig,
    datasets: DatasetBundle | None = None,
    *,
    normalization: NormalizationStats | None = None,
    include_validation: bool = True,
    include_test: bool = True,
) -> tuple[DatasetBundle, LoaderBundle]:
    """Build reproducibly seeded data loaders for every configured split."""

    if datasets is not None and normalization is not None:
        raise ValueError("normalization cannot be supplied with an existing dataset bundle")
    datasets = datasets or build_dataset_splits(
        config,
        normalization=normalization,
        include_validation=include_validation,
        include_test=include_test,
    )
    pin_memory = config.training.device != "cpu" and torch.cuda.is_available()
    train = _loader(
        datasets.train,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
        seed=config.training.seed,
        pin_memory=pin_memory,
    )
    validation = (
        None
        if datasets.validation is None
        else _loader(
            datasets.validation,
            batch_size=config.training.validation_batch_size,
            num_workers=config.training.num_workers,
            shuffle=False,
            seed=config.training.seed + 1,
            pin_memory=pin_memory,
        )
    )
    test = (
        None
        if datasets.test is None
        else _loader(
            datasets.test,
            batch_size=config.evaluation.batch_size,
            num_workers=config.evaluation.num_workers,
            shuffle=False,
            seed=config.training.seed + 2,
            pin_memory=pin_memory,
        )
    )
    return datasets, LoaderBundle(train, validation, test)


def model_statistics(model: torch.nn.Module) -> dict[str, int | float]:
    """Return parameter counts and approximate parameter storage."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    bytes_total = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    return {
        "parameters": total,
        "trainable_parameters": trainable,
        "parameter_megabytes": bytes_total / 1024**2,
    }


__all__ = [
    "LoaderBundle",
    "build_loaders",
    "build_model",
    "build_trainer_settings",
    "model_statistics",
]
