from __future__ import annotations

import torch
from torch import nn

from ptunet.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path) -> None:
    source = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(source.parameters(), lr=1.0e-3)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        source,
        optimizer=optimizer,
        epoch=4,
        global_step=19,
        best_metric=0.125,
        best_epoch=3,
        patience_counter=1,
        normalization={"mean": 2.0, "std": 3.0, "count": 7},
        experiment={"seed": 7},
        history=[{"epoch": 3, "improved": True}],
        best_model_state_dict=source.state_dict(),
        data_loader_state=torch.Generator().manual_seed(9).get_state(),
    )
    restored = nn.Linear(3, 2)

    metadata = load_checkpoint(path, restored)

    for expected, actual in zip(source.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)
    assert metadata.epoch == 4
    assert metadata.global_step == 19
    assert metadata.best_metric == 0.125
    assert metadata.best_epoch == 3
    assert metadata.patience_counter == 1
    assert metadata.normalization == {"mean": 2.0, "std": 3.0, "count": 7}
    assert metadata.history == ({"epoch": 3, "improved": True},)
    assert metadata.best_model_state_dict is not None
    assert metadata.data_loader_state is not None


def test_checkpoint_rejects_unversioned_state_dict(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    model = nn.Linear(2, 2)
    torch.save(model.state_dict(), path)

    try:
        load_checkpoint(path, model)
    except ValueError as error:
        assert "Not a ptunet checkpoint" in str(error)
    else:
        raise AssertionError("Expected an unversioned checkpoint to be rejected")
