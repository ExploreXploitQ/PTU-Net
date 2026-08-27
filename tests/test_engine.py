from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ptunet.checkpoint import load_checkpoint, save_checkpoint
from ptunet.engine import EpochRecord, Trainer, TrainerSettings, _parameter_groups
from ptunet.losses import LossWeights


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs).squeeze(1)


class GroupingProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.global_baseline_logits = nn.Parameter(torch.zeros(3))
        self.baseline_mix_logit = nn.Parameter(torch.zeros(()))
        self.baseline_gate = nn.Conv2d(3, 3, kernel_size=1)


def test_trainer_writes_best_and_last_checkpoints(tmp_path) -> None:
    torch.manual_seed(5)
    inputs = torch.randn(8, 3, 4, 4)
    targets = 0.5 * inputs[:, 0] - 0.2 * inputs[:, 1]
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    settings = TrainerSettings(
        epochs=2,
        learning_rate=1.0e-2,
        early_stopping_patience=2,
        baseline_warmup_epochs=0,
        mixed_precision=False,
        device="cpu",
    )
    trainer = Trainer(TinyRegressor(), settings, tmp_path)

    result = trainer.fit(loader, loader)

    assert result.epochs_completed == 2
    assert result.best_epoch >= 0
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()


def test_explicit_unavailable_cuda_is_rejected(tmp_path) -> None:
    if torch.cuda.is_available():
        return
    settings = TrainerSettings(epochs=1, device="cuda")

    try:
        Trainer(TinyRegressor(), settings, tmp_path)
    except RuntimeError as error:
        assert "CUDA" in str(error)
    else:
        raise AssertionError("Expected unavailable CUDA to be rejected")


def test_only_baseline_logits_receive_warmup_parameter_group() -> None:
    model = GroupingProbe()
    groups = _parameter_groups(model, learning_rate=1.0e-3, weight_decay=0.1)
    assignment = {
        id(parameter): str(group["group_name"]) for group in groups for parameter in group["params"]
    }

    assert assignment[id(model.global_baseline_logits)] == "baseline"
    assert assignment[id(model.baseline_mix_logit)] == "baseline"
    assert assignment[id(model.baseline_gate.weight)] == "main"
    assert assignment[id(model.baseline_gate.bias)] == "no_decay"


def test_incomplete_accumulation_group_matches_sample_weighted_full_batch(tmp_path) -> None:
    torch.manual_seed(3)
    inputs = torch.randn(5, 3, 2, 2)
    targets = torch.randn(5, 2, 2)
    initial = TinyRegressor()
    accumulated_model = copy.deepcopy(initial)
    full_batch_model = copy.deepcopy(initial)
    common = {
        "epochs": 1,
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0e6,
        "baseline_warmup_epochs": 0,
        "mixed_precision": False,
        "device": "cpu",
        "loss": LossWeights(mse=1.0, charbonnier=0.0, correction=0.0),
    }
    accumulated = Trainer(
        accumulated_model,
        TrainerSettings(gradient_accumulation_steps=3, **common),
        tmp_path / "accumulated",
    )
    full_batch = Trainer(
        full_batch_model,
        TrainerSettings(gradient_accumulation_steps=1, **common),
        tmp_path / "full",
    )
    accumulated.optimizer = torch.optim.SGD(accumulated.model.parameters(), lr=0.05)
    full_batch.optimizer = torch.optim.SGD(full_batch.model.parameters(), lr=0.05)

    accumulated._run_epoch(DataLoader(TensorDataset(inputs, targets), batch_size=2), training=True)
    full_batch._run_epoch(DataLoader(TensorDataset(inputs, targets), batch_size=5), training=True)

    for actual, expected in zip(
        accumulated.model.parameters(), full_batch.model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-7)


def test_resume_restores_history_patience_and_best_model_in_new_directory(tmp_path) -> None:
    source = TinyRegressor()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.zero_()
    best_state = {name: torch.ones_like(value) for name, value in source.state_dict().items()}
    history = [
        EpochRecord(0, {"loss": 0.3}, {"loss": 0.2}, [0.01], 1.0, True),
        EpochRecord(1, {"loss": 0.3}, {"loss": 0.25}, [0.01], 1.0, False),
    ]
    checkpoint = tmp_path / "source" / "last.pt"
    save_checkpoint(
        checkpoint,
        source,
        epoch=1,
        global_step=4,
        best_metric=0.2,
        best_epoch=0,
        patience_counter=1,
        history=[record.__dict__ for record in history],
        best_model_state_dict=best_state,
    )
    settings = TrainerSettings(
        epochs=2,
        early_stopping_patience=3,
        baseline_warmup_epochs=0,
        mixed_precision=False,
        device="cpu",
    )
    output = tmp_path / "continued"
    trainer = Trainer(TinyRegressor(), settings, output)
    trainer.resume(checkpoint)
    inputs = torch.zeros(1, 3, 2, 2)
    targets = torch.zeros(1, 2, 2)

    result = trainer.fit(
        DataLoader(TensorDataset(inputs, targets)), DataLoader(TensorDataset(inputs, targets))
    )

    assert result.epochs_completed == 2
    assert result.best_epoch == 0
    assert result.global_step == 4
    assert len(result.history) == 2
    assert (output / "best.pt").is_file()
    assert (output / "last.pt").is_file()
    for value in trainer.model.state_dict().values():
        torch.testing.assert_close(value, torch.ones_like(value))
    last_metadata = load_checkpoint(output / "last.pt", TinyRegressor())
    assert last_metadata.best_epoch == 0
    assert last_metadata.patience_counter == 1
    assert len(last_metadata.history) == 2
