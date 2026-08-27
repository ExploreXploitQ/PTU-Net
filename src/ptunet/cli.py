"""Command line interface for reproducible PTU-Net experiments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from ptunet import __version__
from ptunet.config import ConfigError, ExperimentConfig, load_config


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted configuration field; may be repeated",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptunet",
        description="Train and evaluate PTU-Net scientific field reconstruction models.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train one configured experiment")
    _add_config_arguments(train)
    train.add_argument("--output", type=Path, help="Explicit new run directory")
    train.add_argument("--resume", type=Path, help="Resume a versioned checkpoint")

    evaluate = commands.add_parser("evaluate", help="Reconstruct and score a data split")
    _add_config_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "validation", "test"), default="test")
    evaluate.add_argument("--output", type=Path, help="Evaluation artifact directory")

    inspect = commands.add_parser("inspect-data", help="Validate files and report split sizes")
    _add_config_arguments(inspect)

    summary = commands.add_parser("model-summary", help="Report architecture and parameter count")
    _add_config_arguments(summary)

    resolved = commands.add_parser("show-config", help="Print the fully resolved configuration")
    _add_config_arguments(resolved)

    migrate = commands.add_parser(
        "migrate-checkpoint", help="Convert a format-version-2 prototype checkpoint"
    )
    _add_config_arguments(migrate)
    migrate.add_argument("--legacy", type=Path, required=True)
    migrate.add_argument("--output", type=Path, required=True)
    migrate.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing or unexpected model parameters during migration",
    )
    return parser


def _config(args: argparse.Namespace) -> ExperimentConfig:
    return load_config(args.config, overrides=args.set)


def _train(args: argparse.Namespace) -> int:
    from ptunet.experiment import train_experiment

    run = train_experiment(
        _config(args), output_directory=args.output, resume_checkpoint=args.resume
    )
    print(f"run_directory={run.directory}")
    print(f"best_checkpoint={run.best_checkpoint}")
    print(f"best_validation_loss={run.result.best_validation_loss:.8g}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from ptunet.experiment import evaluate_experiment

    run = evaluate_experiment(
        _config(args),
        args.checkpoint,
        split=args.split,
        output_directory=args.output,
    )
    print(f"evaluation_directory={run.directory}")
    print(f"metrics={run.metrics_csv}")
    return 0


def _inspect_data(args: argparse.Namespace) -> int:
    from ptunet.data import build_dataset_splits

    config = _config(args)
    bundle = build_dataset_splits(config)
    payload = {
        "data_root": str(config.paths.data_root),
        "normalization": bundle.normalization.to_dict(),
        "patches": {
            "train": len(bundle.train),
            "validation": None if bundle.validation is None else len(bundle.validation),
            "test": None if bundle.test is None else len(bundle.test),
        },
        "datasets": [
            {
                "name": spec.name,
                "shape": list(spec.shape),
                "train_timesteps": len(spec.train_timesteps),
                "validation_timesteps": len(spec.validation_timesteps),
                "test_timesteps": len(spec.test_timesteps),
            }
            for spec in config.datasets
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _model_summary(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from ptunet.factory import build_model, model_statistics

    config = _config(args)
    model = build_model(config.model)
    payload: dict[str, Any] = {
        "model": asdict(model.config),
        **model_statistics(model),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _show_config(args: argparse.Namespace) -> int:
    print(yaml.safe_dump(_config(args).to_dict(), sort_keys=False), end="")
    return 0


def _migrate_checkpoint(args: argparse.Namespace) -> int:
    import torch

    from ptunet.checkpoint import save_checkpoint
    from ptunet.data import compute_training_normalization
    from ptunet.factory import build_model

    config = _config(args)
    try:
        legacy = torch.load(args.legacy, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch
        legacy = torch.load(args.legacy, map_location="cpu")
    if not isinstance(legacy, dict):
        raise ValueError("Legacy checkpoint must contain a dictionary")
    model = build_model(config.model)
    report = model.load_legacy_state_dict(legacy, strict=not args.allow_partial)
    normalization = compute_training_normalization(
        config.datasets,
        config.paths.data_root,
        chunk_elements=config.training.normalization_chunk_elements,
    )
    save_checkpoint(
        args.output,
        model,
        epoch=-1,
        global_step=0,
        best_metric=None,
        normalization=normalization.to_dict(),
        experiment=config.to_dict(),
    )
    print(f"checkpoint={args.output.resolve()}")
    print(f"missing_keys={list(report.missing_keys)}")
    print(f"unexpected_keys={list(report.unexpected_keys)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    actions = {
        "train": _train,
        "evaluate": _evaluate,
        "inspect-data": _inspect_data,
        "model-summary": _model_summary,
        "show-config": _show_config,
        "migrate-checkpoint": _migrate_checkpoint,
    }
    try:
        return actions[args.command](args)
    except (ConfigError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ptunet: error: {error}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
