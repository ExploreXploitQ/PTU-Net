from __future__ import annotations

from pathlib import Path

import yaml

from ptunet.cli import main

REPOSITORY = Path(__file__).resolve().parents[1]
SYNTHETIC_CONFIG = REPOSITORY / "configs" / "synthetic.yaml"


def test_show_config_applies_typed_override(capsys) -> None:
    status = main(
        [
            "show-config",
            "--config",
            str(SYNTHETIC_CONFIG),
            "--set",
            "training.epochs=7",
        ]
    )

    document = yaml.safe_load(capsys.readouterr().out)
    assert status == 0
    assert document["training"]["epochs"] == 7
    assert Path(document["paths"]["data_root"]).is_absolute()


def test_model_summary_does_not_require_dataset_files(capsys) -> None:
    status = main(["model-summary", "--config", str(SYNTHETIC_CONFIG)])
    output = capsys.readouterr().out

    assert status == 0
    assert '"parameters"' in output
    assert '"patch_size": 16' in output


def test_unknown_override_returns_clean_error(capsys) -> None:
    status = main(
        [
            "show-config",
            "--config",
            str(SYNTHETIC_CONFIG),
            "--set",
            "training.not_a_field=1",
        ]
    )

    assert status == 2
    assert "unknown key" in capsys.readouterr().err
