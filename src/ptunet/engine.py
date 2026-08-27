"""Training and validation engine for PTU-Net experiments."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ptunet.checkpoint import load_checkpoint, save_checkpoint
from ptunet.losses import CompositeReconstructionLoss, LossWeights
from ptunet.tracking import NullTracker, Tracker


@dataclass(frozen=True)
class TrainerSettings:
    """Runtime settings consumed by :class:`Trainer`."""

    epochs: int = 300
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-3
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    minimum_learning_rate: float = 1.0e-7
    early_stopping_patience: int = 15
    minimum_delta: float = 1.0e-6
    gradient_clip_norm: float = 0.5
    gradient_accumulation_steps: int = 1
    baseline_warmup_epochs: int = 5
    mixed_precision: bool = True
    device: str = "auto"
    loss: LossWeights = field(default_factory=LossWeights)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0 or self.minimum_learning_rate < 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be between zero and one")
        if self.scheduler_patience < 0 or self.early_stopping_patience < 1:
            raise ValueError("patience values are invalid")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")


@dataclass
class EpochRecord:
    epoch: int
    train: dict[str, float]
    validation: dict[str, float]
    learning_rates: list[float]
    duration_seconds: float
    improved: bool


@dataclass
class TrainingResult:
    best_validation_loss: float
    best_epoch: int
    epochs_completed: int
    global_step: int
    history: list[EpochRecord]
    stopped_early: bool


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, ``cuda``, or an explicit CUDA device."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
    return device


def _make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # pragma: no cover - older PyTorch API
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover


def _unpack_batch(batch: Any) -> tuple[Tensor, Tensor]:
    if isinstance(batch, Mapping):
        return batch["inputs"], batch["target"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("A batch must be a mapping or an (inputs, target) pair")


def _model_prediction(model: nn.Module, inputs: Tensor) -> tuple[Tensor, Tensor | None]:
    forward_diagnostics = getattr(model, "forward_with_diagnostics", None)
    if callable(forward_diagnostics):
        prediction, diagnostics = forward_diagnostics(inputs)
        return prediction, getattr(diagnostics, "correction_map", None)
    return model(inputs), getattr(model, "last_correction_map", None)


def _cpu_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _parameter_groups(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> list[dict[str, Any]]:
    groups: dict[str, list[nn.Parameter]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        parameter_name = lowered.rsplit(".", 1)[-1]
        if parameter_name in {
            "global_baseline_logits",
            "baseline_mix_logit",
            "compressor_weights",
            "baseline_mix",
        }:
            key = "baseline"
        elif "unet" in lowered or "refinement" in lowered:
            key = "refinement"
        elif "correction" in lowered:
            key = "correction"
        elif parameter.ndim <= 1 or name.endswith(".bias"):
            key = "no_decay"
        else:
            key = "main"

        groups[key].append(parameter)

    specifications = {
        "main": (learning_rate, weight_decay),
        "no_decay": (learning_rate, 0.0),
        "baseline": (learning_rate * 0.5, 0.0),
        "correction": (learning_rate, 0.0),
        "refinement": (learning_rate * 2.0, 0.0),
    }
    return [
        {
            "params": groups[name],
            "lr": specifications[name][0],
            "initial_lr": specifications[name][0],
            "weight_decay": specifications[name][1],
            "group_name": name,
        }
        for name in specifications
        if groups[name]
    ]


class Trainer:
    """Stateful trainer with early stopping and versioned checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        settings: TrainerSettings,
        output_directory: str | Path,
        *,
        tracker: Tracker | None = None,
        experiment_config: Any = None,
        normalization: Mapping[str, float] | None = None,
    ) -> None:
        self.settings = settings
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(settings.device)
        self.model = model.to(self.device)
        self.objective = CompositeReconstructionLoss(settings.loss)
        self.tracker = tracker or NullTracker()
        self.experiment_config = experiment_config
        self.normalization = None if normalization is None else dict(normalization)
        self.optimizer = torch.optim.AdamW(
            _parameter_groups(model, settings.learning_rate, settings.weight_decay),
            lr=settings.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=settings.scheduler_factor,
            patience=settings.scheduler_patience,
            min_lr=settings.minimum_learning_rate,
        )
        self.use_amp = settings.mixed_precision and self.device.type == "cuda"
        self.scaler = _make_grad_scaler(self.use_amp)
        self.global_step = 0
        self.start_epoch = 0
        self.best_validation_loss = math.inf
        self.best_epoch = -1
        self.patience_counter = 0
        self.history: list[EpochRecord] = []
        self._best_state: dict[str, Tensor] | None = None
        self._data_loader_state: Tensor | None = None
        self._resumed_stopped_early = False

    def resume(self, checkpoint: str | Path) -> None:
        """Resume model, optimizer, scheduler, and scaler state."""

        metadata = load_checkpoint(
            checkpoint,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
            restore_rng=True,
        )
        self.start_epoch = metadata.epoch + 1
        self.global_step = metadata.global_step
        if metadata.best_metric is not None:
            self.best_validation_loss = metadata.best_metric
        self.best_epoch = metadata.best_epoch
        self.patience_counter = metadata.patience_counter
        self.history = [EpochRecord(**record) for record in metadata.history]
        self._best_state = (
            None
            if metadata.best_model_state_dict is None
            else _cpu_state_dict(metadata.best_model_state_dict)
        )
        self._data_loader_state = metadata.data_loader_state
        self._resumed_stopped_early = metadata.stopped_early
        if metadata.normalization is not None:
            self.normalization = metadata.normalization

        if self.best_epoch < 0:
            improved_epochs = [record.epoch for record in self.history if record.improved]
            if improved_epochs:
                self.best_epoch = improved_epochs[-1]
        if metadata.format_version == 1 and self.history:
            self.patience_counter = 0
            for record in reversed(self.history):
                if record.improved:
                    break
                self.patience_counter += 1
        if self._best_state is None and (
            metadata.is_best_checkpoint
            or (metadata.format_version == 1 and Path(checkpoint).name == "best.pt")
        ):
            self._best_state = _cpu_state_dict(self.model.state_dict())
            if self.best_epoch < 0:
                self.best_epoch = metadata.epoch
        if self._best_state is None and metadata.format_version == 1:
            best_path = Path(checkpoint).with_name("best.pt")
            if best_path.is_file() and best_path != Path(checkpoint):
                resumed_state = _cpu_state_dict(self.model.state_dict())
                try:
                    best_metadata = load_checkpoint(
                        best_path,
                        self.model,
                        map_location=self.device,
                    )
                    self._best_state = _cpu_state_dict(self.model.state_dict())
                    if self.best_epoch < 0:
                        self.best_epoch = best_metadata.epoch
                finally:
                    self.model.load_state_dict(resumed_state)
        if (
            metadata.format_version >= 2
            and metadata.best_metric is not None
            and self._best_state is None
            and not metadata.is_best_checkpoint
        ):
            raise ValueError("Checkpoint records a best metric but has no best model state")

    def _set_baseline_learning_rate(self, epoch: int) -> None:
        for group in self.optimizer.param_groups:
            if group.get("group_name") == "baseline":
                initial = float(group.get("initial_lr", self.settings.learning_rate * 0.5))
                if epoch < self.settings.baseline_warmup_epochs:
                    group["lr"] = 0.0
                elif epoch == self.settings.baseline_warmup_epochs and group["lr"] == 0.0:
                    group["lr"] = initial

    def _run_epoch(self, loader: DataLoader[Any], training: bool) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = defaultdict(float)
        sample_count = 0
        accumulation = self.settings.gradient_accumulation_steps
        accumulated_samples = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(loader):
            inputs, targets = _unpack_batch(batch)
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = int(inputs.shape[0])

            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    prediction, correction = _model_prediction(self.model, inputs)
                    output = self.objective(prediction, targets, correction)
                    scaled_loss = output.total * batch_size

                if training:
                    self.scaler.scale(scaled_loss).backward()
                    accumulated_samples += batch_size
                    is_update = (batch_index + 1) % accumulation == 0 or (
                        batch_index + 1 == len(loader)
                    )
                    if is_update:
                        self.scaler.unscale_(self.optimizer)
                        for parameter in self.model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(accumulated_samples)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.settings.gradient_clip_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        accumulated_samples = 0
                    self.global_step += 1

            for name, value in output.components.items():
                totals[name] += value * batch_size
            batch_mae = float(torch.mean(torch.abs(prediction - targets)).detach())
            totals["mae"] += batch_mae * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise RuntimeError("The data loader produced no batches")
        return {name: value / sample_count for name, value in totals.items()}

    def _write_checkpoint(
        self,
        name: str,
        epoch: int,
        *,
        model_state_dict: Mapping[str, Tensor] | None = None,
        stopped_early: bool = False,
    ) -> None:
        save_checkpoint(
            self.output_directory / name,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            global_step=self.global_step,
            best_metric=(
                None if not math.isfinite(self.best_validation_loss) else self.best_validation_loss
            ),
            best_epoch=self.best_epoch,
            patience_counter=self.patience_counter,
            stopped_early=stopped_early,
            is_best_checkpoint=name == "best.pt",
            normalization=self.normalization,
            experiment=self.experiment_config,
            history=[asdict(record) for record in self.history],
            best_model_state_dict=None if name == "best.pt" else self._best_state,
            model_state_dict=model_state_dict,
            data_loader_state=self._data_loader_state,
        )

    def fit(
        self,
        train_loader: DataLoader[Any],
        validation_loader: DataLoader[Any],
    ) -> TrainingResult:
        """Train until the epoch limit or early-stopping criterion is reached."""

        history = list(self.history)
        patience_counter = self.patience_counter
        best_state = self._best_state
        stopped_early = False
        epochs_run = 0

        loader_generator = getattr(train_loader, "generator", None)
        if self._data_loader_state is not None and isinstance(loader_generator, torch.Generator):
            loader_generator.set_state(self._data_loader_state)

        try:
            for epoch in range(self.start_epoch, self.settings.epochs):
                started = time.perf_counter()
                self._set_baseline_learning_rate(epoch)
                train_metrics = self._run_epoch(train_loader, training=True)
                validation_metrics = self._run_epoch(validation_loader, training=False)
                validation_loss = validation_metrics["loss"]
                improved = validation_loss < self.best_validation_loss - self.settings.minimum_delta
                if improved:
                    self.best_validation_loss = validation_loss
                    self.best_epoch = epoch
                    patience_counter = 0
                    best_state = _cpu_state_dict(self.model.state_dict())
                else:
                    patience_counter += 1

                self.scheduler.step(validation_loss)
                learning_rates = [float(group["lr"]) for group in self.optimizer.param_groups]
                record = EpochRecord(
                    epoch=epoch,
                    train=train_metrics,
                    validation=validation_metrics,
                    learning_rates=learning_rates,
                    duration_seconds=time.perf_counter() - started,
                    improved=improved,
                )
                history.append(record)
                epochs_run += 1
                self.history = history
                self.patience_counter = patience_counter
                self._best_state = best_state
                if isinstance(loader_generator, torch.Generator):
                    self._data_loader_state = loader_generator.get_state()
                self.tracker.log(
                    {
                        **{f"train/{key}": value for key, value in train_metrics.items()},
                        **{f"validation/{key}": value for key, value in validation_metrics.items()},
                        "epoch": epoch,
                        "epoch_seconds": record.duration_seconds,
                        "learning_rate": learning_rates[0],
                        "patience": patience_counter,
                    },
                    step=epoch,
                )
                will_stop = patience_counter >= self.settings.early_stopping_patience
                self._write_checkpoint("last.pt", epoch, stopped_early=will_stop)
                if improved:
                    assert best_state is not None
                    self._write_checkpoint("best.pt", epoch, model_state_dict=best_state)

                print(
                    f"epoch={epoch + 1}/{self.settings.epochs} "
                    f"train_loss={train_metrics['loss']:.6g} "
                    f"val_loss={validation_loss:.6g} "
                    f"lr={learning_rates[0]:.3g}"
                )
                if will_stop:
                    stopped_early = True
                    break

            last_path = self.output_directory / "last.pt"
            if not last_path.exists():
                self._write_checkpoint(
                    "last.pt",
                    self.start_epoch - 1,
                    model_state_dict=self.model.state_dict(),
                    stopped_early=self._resumed_stopped_early,
                )
            if best_state is not None:
                self.model.load_state_dict(best_state)
                best_path = self.output_directory / "best.pt"
                if not best_path.exists():
                    self._write_checkpoint(
                        "best.pt",
                        self.best_epoch,
                        model_state_dict=best_state,
                    )
            completed = len(history)
            if epochs_run == 0:
                stopped_early = self._resumed_stopped_early
            result = TrainingResult(
                best_validation_loss=self.best_validation_loss,
                best_epoch=self.best_epoch,
                epochs_completed=completed,
                global_step=self.global_step,
                history=history,
                stopped_early=stopped_early,
            )
            self.tracker.update_summary(
                {
                    "best_validation_loss": result.best_validation_loss,
                    "best_epoch": result.best_epoch,
                    "epochs_completed": completed,
                    "global_step": result.global_step,
                    "stopped_early": stopped_early,
                    "trainer": asdict(self.settings),
                }
            )
            return result
        finally:
            self.tracker.close()


@torch.inference_mode()
def predict_batches(
    model: nn.Module,
    batches: Iterable[Any],
    device: str | torch.device = "auto",
) -> list[Tensor]:
    """Run inference over batches and return CPU prediction tensors."""

    resolved = resolve_device(device) if isinstance(device, str) else device
    model = model.to(resolved).eval()
    predictions = []
    for batch in batches:
        inputs, _ = _unpack_batch(batch)
        predictions.append(model(inputs.to(resolved)).detach().cpu())
    return predictions


__all__ = [
    "EpochRecord",
    "Trainer",
    "TrainerSettings",
    "TrainingResult",
    "predict_batches",
    "resolve_device",
]
