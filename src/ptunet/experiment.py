"""End-to-end training, evaluation, and experiment artifact management."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ptunet.checkpoint import CheckpointMetadata, load_checkpoint
from ptunet.config import ExperimentConfig
from ptunet.data import (
    DatasetBundle,
    NormalizationStats,
    resolve_temporal_window_from_files,
)
from ptunet.engine import EpochRecord, Trainer, TrainingResult
from ptunet.factory import build_loaders, build_model, build_trainer_settings, model_statistics
from ptunet.inference import reconstruct_field
from ptunet.io import (
    CompressorResolver,
    compressed_path,
    open_float32_memmap,
    original_path,
    reconstruction_path,
    write_float32_atomic,
)
from ptunet.metrics import compute_metrics
from ptunet.reproducibility import seed_everything, write_environment
from ptunet.tracking import CompositeTracker, JsonlTracker, Tracker, WandbTracker


@dataclass(frozen=True)
class TrainingRun:
    directory: Path
    best_checkpoint: Path
    last_checkpoint: Path
    result: TrainingResult
    datasets: DatasetBundle


@dataclass(frozen=True)
class EvaluationRun:
    directory: Path
    metrics_csv: Path
    records: tuple[dict[str, Any], ...]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_directory(config: ExperimentConfig, requested: str | Path | None = None) -> Path:
    """Create a new run directory without overwriting an earlier experiment."""

    if requested is None:
        base = config.paths.output_root / config.name
        candidate = base / _timestamp()
        suffix = 1
        while candidate.exists():
            candidate = base / f"{_timestamp()}-{suffix}"
            suffix += 1
    else:
        candidate = Path(requested).expanduser().resolve()
        if candidate.exists() and any(candidate.iterdir()):
            raise FileExistsError(
                f"Run directory is not empty: {candidate}. Choose a new path or resume."
            )
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                return None
            return "Infinity" if value > 0 else "-Infinity"
        if isinstance(value, Path):
            return str(value)
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_tracker(config: ExperimentConfig, run_directory: Path) -> Tracker:
    trackers: list[Tracker] = [JsonlTracker(run_directory / "metrics.jsonl")]
    if config.tracking.enabled:
        trackers.append(
            WandbTracker(
                project=config.tracking.project,
                name=config.tracking.run_name or config.name,
                config=config.to_dict(),
                entity=config.tracking.entity,
                tags=config.tracking.tags,
            )
        )
    return CompositeTracker(trackers)


def _write_history(path: Path, result: TrainingResult) -> None:
    fieldnames = [
        "epoch",
        "duration_seconds",
        "improved",
        "learning_rate",
        "train_loss",
        "train_mse",
        "train_charbonnier",
        "train_correction",
        "train_gradient",
        "train_spectral",
        "train_mae",
        "validation_loss",
        "validation_mse",
        "validation_charbonnier",
        "validation_correction",
        "validation_gradient",
        "validation_spectral",
        "validation_mae",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in result.history:
            row: dict[str, Any] = {
                "epoch": record.epoch,
                "duration_seconds": record.duration_seconds,
                "improved": record.improved,
                "learning_rate": record.learning_rates[0],
            }
            row.update({f"train_{key}": value for key, value in record.train.items()})
            row.update({f"validation_{key}": value for key, value in record.validation.items()})
            writer.writerow(row)
    temporary.replace(path)


def _seed_local_history(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Rebuild local epoch events when a resumed run uses a new directory."""

    if path.exists():
        return
    tracker = JsonlTracker(path)
    patience = 0
    for values in records:
        record = EpochRecord(**dict(values))
        patience = 0 if record.improved else patience + 1
        tracker.log(
            {
                **{f"train/{key}": value for key, value in record.train.items()},
                **{f"validation/{key}": value for key, value in record.validation.items()},
                "epoch": record.epoch,
                "epoch_seconds": record.duration_seconds,
                "learning_rate": record.learning_rates[0],
                "patience": patience,
            },
            step=record.epoch,
        )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _mapping_mismatches(
    checkpoint_values: Mapping[str, Any],
    current_values: Mapping[str, Any],
) -> list[str]:
    mismatches = []
    for key in sorted(set(checkpoint_values) | set(current_values)):
        if (
            key not in checkpoint_values
            or key not in current_values
            or _canonical_value(checkpoint_values[key]) != _canonical_value(current_values[key])
        ):
            mismatches.append(key)
    return mismatches


def _validate_checkpoint_model_config(
    metadata: CheckpointMetadata,
    config: ExperimentConfig,
) -> None:
    stored = metadata.experiment.get("model")
    if stored is None:
        raise ValueError(
            "Checkpoint does not record model configuration metadata; "
            "migrate it before evaluation or resume"
        )
    if not isinstance(stored, Mapping):
        raise ValueError("Checkpoint experiment.model metadata is malformed")
    mismatches = _mapping_mismatches(stored, asdict(config.model))
    if mismatches:
        fields = ", ".join(f"model.{name}" for name in mismatches)
        raise ValueError(
            "Configured model does not match the checkpoint; behavior-changing "
            f"field(s): {fields}. Use the checkpoint's model settings."
        )


def _validate_resume_config(metadata: CheckpointMetadata, config: ExperimentConfig) -> None:
    _validate_checkpoint_model_config(metadata, config)
    stored = metadata.experiment.get("training")
    if stored is None:
        return
    if not isinstance(stored, Mapping):
        raise ValueError("Checkpoint experiment.training metadata is malformed")

    current = asdict(config.training)
    safe_overrides = {
        "batch_size",
        "device",
        "epochs",
        "normalization_chunk_elements",
        "num_workers",
        "validation_batch_size",
    }
    stateful_checkpoint = {key: value for key, value in stored.items() if key not in safe_overrides}
    stateful_current = {key: value for key, value in current.items() if key not in safe_overrides}
    mismatches = _mapping_mismatches(stateful_checkpoint, stateful_current)
    if mismatches:
        fields = ", ".join(f"training.{name}" for name in mismatches)
        raise ValueError(
            "Resume configuration changes stateful training behavior in field(s): "
            f"{fields}. Start a new run for those changes."
        )

    checkpoint_epochs = stored.get("epochs")
    if isinstance(checkpoint_epochs, int) and config.training.epochs < checkpoint_epochs:
        raise ValueError(
            "training.epochs may stay unchanged or increase when resuming; "
            f"checkpoint={checkpoint_epochs}, configured={config.training.epochs}"
        )


def train_experiment(
    config: ExperimentConfig,
    *,
    output_directory: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> TrainingRun:
    """Train one configured experiment and persist a complete local run record."""

    seed_everything(config.training.seed)
    model = build_model(config.model)
    checkpoint_path: Path | None = None
    resume_metadata: CheckpointMetadata | None = None
    resume_normalization: NormalizationStats | None = None
    if resume_checkpoint is not None:
        checkpoint_path = Path(resume_checkpoint).expanduser().resolve()
        resume_metadata = load_checkpoint(checkpoint_path, model, map_location="cpu")
        _validate_resume_config(resume_metadata, config)
        resume_normalization = _normalization_from_checkpoint(
            resume_metadata.normalization,
            config,
        )

    datasets, loaders = build_loaders(
        config,
        normalization=resume_normalization,
        include_test=False,
    )
    if loaders.validation is None or len(datasets.validation or ()) == 0:
        raise ValueError("Training requires at least one validation timestep")

    if resume_checkpoint is not None and output_directory is None:
        assert checkpoint_path is not None
        run_directory = checkpoint_path.parent
        run_directory.mkdir(parents=True, exist_ok=True)
    else:
        run_directory = create_run_directory(config, output_directory)

    _write_yaml(run_directory / "config.resolved.yaml", config.to_dict())
    write_environment(run_directory / "environment.json")
    _write_json(
        run_directory / "model.json",
        {"architecture": asdict(model.config), **model_statistics(model)},
    )
    normalization = datasets.normalization.to_dict()
    _write_json(run_directory / "normalization.json", normalization)
    if resume_metadata is not None:
        _seed_local_history(run_directory / "metrics.jsonl", resume_metadata.history)

    trainer = Trainer(
        model,
        build_trainer_settings(config.training),
        run_directory,
        tracker=_build_tracker(config, run_directory),
        experiment_config=config.to_dict(),
        normalization=normalization,
    )
    if resume_checkpoint is not None:
        assert checkpoint_path is not None
        trainer.resume(checkpoint_path)
        assert resume_metadata is not None
        _write_json(
            run_directory / "resume.json",
            {
                "checkpoint": checkpoint_path,
                "checkpoint_format_version": resume_metadata.format_version,
                "checkpoint_epoch": resume_metadata.epoch,
                "next_epoch": trainer.start_epoch,
                "optimizer_learning_rates": [
                    float(group["lr"]) for group in trainer.optimizer.param_groups
                ],
                "normalization_source": (
                    "checkpoint" if resume_metadata.normalization is not None else "training_data"
                ),
            },
        )
    result = trainer.fit(loaders.train, loaders.validation)
    _write_history(run_directory / "history.csv", result)
    _write_json(
        run_directory / "training_summary.json",
        {
            "best_validation_loss": result.best_validation_loss,
            "best_epoch": result.best_epoch,
            "epochs_completed": result.epochs_completed,
            "global_step": result.global_step,
            "stopped_early": result.stopped_early,
        },
    )
    return TrainingRun(
        run_directory,
        run_directory / "best.pt",
        run_directory / "last.pt",
        result,
        datasets,
    )


def _normalization_from_checkpoint(
    checkpoint_normalization: Mapping[str, float | int] | None,
    _config: ExperimentConfig,
) -> NormalizationStats:
    if checkpoint_normalization is not None:
        try:
            return NormalizationStats.from_dict(checkpoint_normalization)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Checkpoint normalization metadata is malformed") from error
    raise ValueError(
        "Checkpoint does not contain normalization metadata; migrate it before evaluation or resume"
    )


def _write_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    rows = list(records)
    if not rows:
        raise ValueError("No evaluation records were produced")
    fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_experiment(
    config: ExperimentConfig,
    checkpoint: str | Path,
    *,
    split: str = "test",
    output_directory: str | Path | None = None,
) -> EvaluationRun:
    """Reconstruct and score every center timestep in one configured split."""

    normalized_split = "validation" if split == "val" else split
    if normalized_split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    model = build_model(config.model)
    metadata = load_checkpoint(checkpoint_path, model, map_location="cpu")
    _validate_checkpoint_model_config(metadata, config)
    normalization = _normalization_from_checkpoint(metadata.normalization, config)
    destination = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else checkpoint_path.parent / "evaluation" / normalized_split
    )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Evaluation directory is not empty: {destination}. Choose a new path."
        )
    destination.mkdir(parents=True, exist_ok=True)
    resolver = CompressorResolver(config.compressor_extensions)
    records: list[dict[str, Any]] = []

    for spec in config.datasets:
        for center in spec.timesteps_for_split(normalized_split):
            window = resolve_temporal_window_from_files(
                spec,
                center,
                config.paths.data_root,
                resolver,
                search_radius=config.evaluation.temporal_search_radius,
            )
            temporal_fields = tuple(
                open_float32_memmap(
                    compressed_path(spec, config.paths.data_root, timestep, resolver),
                    spec.shape,
                )
                for timestep in window.timesteps
            )
            reconstruction = reconstruct_field(
                model,
                temporal_fields,
                normalization,
                patch_size=config.model.patch_size,
                stride=config.model.stride,
                batch_size=config.evaluation.batch_size,
                num_workers=config.evaluation.num_workers,
                device=config.training.device,
                input_mode=config.model.input_mode,  # type: ignore[arg-type]
            )
            target = open_float32_memmap(
                original_path(spec, config.paths.data_root, center), spec.shape
            )
            baseline = temporal_fields[1]
            learned = compute_metrics(
                reconstruction.field,
                target,
                data_range=config.evaluation.data_range,
                chunk_elements=config.evaluation.metric_chunk_elements,
            )
            reference = compute_metrics(
                baseline,
                target,
                data_range=config.evaluation.data_range,
                chunk_elements=config.evaluation.metric_chunk_elements,
            )
            output_path = reconstruction_path(spec, destination, center, resolver)
            write_float32_atomic(output_path, reconstruction.field)
            records.append(
                {
                    "dataset": spec.name,
                    "center_timestep": center,
                    "previous_timestep": window.previous,
                    "next_timestep": window.next,
                    "center_compressor": resolver.compressor_for(spec, center),
                    "count": learned.count,
                    "mse": learned.mse,
                    "mae": learned.mae,
                    "psnr_db": learned.psnr,
                    "global_ssim": learned.global_ssim,
                    "data_range": learned.data_range,
                    "baseline_mse": reference.mse,
                    "baseline_mae": reference.mae,
                    "baseline_psnr_db": reference.psnr,
                    "baseline_global_ssim": reference.global_ssim,
                    "psnr_improvement_db": learned.psnr - reference.psnr,
                    "inference_seconds": reconstruction.elapsed_seconds,
                    "patches": reconstruction.patch_count,
                    "reconstruction": str(output_path),
                }
            )
            print(
                f"dataset={spec.name} timestep={center} mse={learned.mse:.6g} "
                f"psnr={learned.psnr:.4g} patches={reconstruction.patch_count}"
            )

    metrics_csv = destination / "metrics.csv"
    _write_records(metrics_csv, records)
    _write_json(
        destination / "evaluation.json",
        {
            "checkpoint": checkpoint_path,
            "split": normalized_split,
            "normalization": normalization.to_dict(),
            "records": records,
        },
    )
    return EvaluationRun(destination, metrics_csv, tuple(records))


__all__ = [
    "EvaluationRun",
    "TrainingRun",
    "create_run_directory",
    "evaluate_experiment",
    "train_experiment",
]
