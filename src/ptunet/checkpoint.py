"""Versioned and atomic PTU-Net checkpoints."""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

CHECKPOINT_FORMAT = "ptunet"
CHECKPOINT_VERSION = 2
SUPPORTED_CHECKPOINT_VERSIONS = (1, CHECKPOINT_VERSION)


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata returned when a checkpoint is restored."""

    format_version: int
    epoch: int
    global_step: int
    best_metric: float | None
    best_epoch: int
    patience_counter: int
    stopped_early: bool
    is_best_checkpoint: bool
    normalization: dict[str, float | int] | None
    experiment: dict[str, Any]
    history: tuple[dict[str, Any], ...]
    best_model_state_dict: dict[str, torch.Tensor] | None
    data_loader_state: torch.Tensor | None


def _plain_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("experiment config must be a dataclass, mapping, or None")


def _model_cuda_device(model: nn.Module) -> torch.device | None:
    tensors = list(model.parameters()) + list(model.buffers())
    for tensor in tensors:
        if tensor.device.type == "cuda":
            return tensor.device
    return None


def _capture_rng_state(model: nn.Module) -> dict[str, Any]:
    numpy_state: Any = np.random.get_state()
    cuda_device = _model_cuda_device(model)
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            None if cuda_device is None else torch.cuda.get_rng_state(cuda_device).cpu()
        ),
    }


def _restore_rng_state(state: Any, model: nn.Module) -> None:
    if not isinstance(state, Mapping):
        return
    python_state = state.get("python")
    if isinstance(python_state, tuple):
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if isinstance(numpy_state, Mapping):
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    torch_cpu = state.get("torch_cpu")
    if isinstance(torch_cpu, torch.Tensor):
        torch.set_rng_state(torch_cpu.cpu())
    torch_cuda = state.get("torch_cuda")
    cuda_device = _model_cuda_device(model)
    if cuda_device is not None and isinstance(torch_cuda, torch.Tensor):
        torch.cuda.set_rng_state(torch_cuda.cpu(), device=cuda_device)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    best_epoch: int = -1,
    patience_counter: int = 0,
    stopped_early: bool = False,
    is_best_checkpoint: bool = False,
    normalization: Mapping[str, float | int] | None = None,
    experiment: Any = None,
    history: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None = None,
    best_model_state_dict: Mapping[str, torch.Tensor] | None = None,
    model_state_dict: Mapping[str, torch.Tensor] | None = None,
    data_loader_state: torch.Tensor | None = None,
) -> None:
    """Write a checkpoint with an atomic rename on the destination filesystem."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved_model_state = model.state_dict() if model_state_dict is None else model_state_dict
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "model_state_dict": dict(saved_model_state),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": None if best_metric is None else float(best_metric),
        "best_epoch": int(best_epoch),
        "patience_counter": int(patience_counter),
        "stopped_early": bool(stopped_early),
        "is_best_checkpoint": bool(is_best_checkpoint),
        "normalization": None if normalization is None else dict(normalization),
        "experiment": _plain_config(experiment),
        "history": [] if history is None else [dict(record) for record in history],
        "best_model_state_dict": (
            None if best_model_state_dict is None else dict(best_model_state_dict)
        ),
        "data_loader_state": data_loader_state,
        "rng_state": _capture_rng_state(model),
    }
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_torch_load(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - PyTorch before weights_only
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = False,
) -> CheckpointMetadata:
    """Restore model and optional training state from a PTU-Net checkpoint."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    payload = _safe_torch_load(source, map_location)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a {CHECKPOINT_FORMAT} checkpoint: {source}")
    version = int(payload.get("format_version", 0))
    if version not in SUPPORTED_CHECKPOINT_VERSIONS:
        supported = ", ".join(str(item) for item in SUPPORTED_CHECKPOINT_VERSIONS)
        raise ValueError(
            f"Unsupported checkpoint format {version}; supported versions: {supported}"
        )
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if restore_rng:
        _restore_rng_state(payload.get("rng_state"), model)
    normalization = payload.get("normalization")
    if normalization is not None and not isinstance(normalization, Mapping):
        raise ValueError(f"Checkpoint normalization metadata is malformed: {source}")
    experiment = payload.get("experiment", {})
    if not isinstance(experiment, Mapping):
        raise ValueError(f"Checkpoint experiment metadata is malformed: {source}")
    history = payload.get("history", [])
    if not isinstance(history, list) or not all(isinstance(item, Mapping) for item in history):
        raise ValueError(f"Checkpoint history is malformed: {source}")
    best_state = payload.get("best_model_state_dict")
    if best_state is not None and not isinstance(best_state, Mapping):
        raise ValueError(f"Checkpoint best model state is malformed: {source}")
    if best_state is not None and not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in best_state.items()
    ):
        raise ValueError(f"Checkpoint best model state is malformed: {source}")
    loader_state = payload.get("data_loader_state")
    if loader_state is not None and not isinstance(loader_state, torch.Tensor):
        raise ValueError(f"Checkpoint data loader state is malformed: {source}")
    return CheckpointMetadata(
        format_version=version,
        epoch=int(payload.get("epoch", -1)),
        global_step=int(payload.get("global_step", 0)),
        best_metric=(None if payload.get("best_metric") is None else float(payload["best_metric"])),
        best_epoch=int(payload.get("best_epoch", -1)),
        patience_counter=int(payload.get("patience_counter", 0)),
        stopped_early=bool(payload.get("stopped_early", False)),
        is_best_checkpoint=bool(payload.get("is_best_checkpoint", False)),
        normalization=(
            None
            if normalization is None
            else {
                str(key): int(value) if str(key) == "count" else float(value)
                for key, value in normalization.items()
            }
        ),
        experiment=dict(experiment),
        history=tuple(dict(item) for item in history),
        best_model_state_dict=(
            None if best_state is None else {str(key): value for key, value in best_state.items()}
        ),
        data_loader_state=None if loader_state is None else loader_state.cpu(),
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "SUPPORTED_CHECKPOINT_VERSIONS",
    "CheckpointMetadata",
    "load_checkpoint",
    "save_checkpoint",
]
