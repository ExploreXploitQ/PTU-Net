# Usage

This guide covers the package workflow without assuming access to the original host or research data. Start with a dedicated Python environment and a YAML file that follows [the configuration guide](configuration.md).

## Install from source

```bash
git clone https://github.com/ExploreXploitQ/PTU-Net.git
cd PTU-Net
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Check the installed entry point and package version:

```bash
ptunet --help
python -c "import ptunet; print(ptunet.__version__)"
```

## Prepare data

PTU-Net expects original fields and already reconstructed compressor outputs. It does not run a compressor. Verify byte order, shape, names, splits, and temporal neighbors against [the data contract](data-contract.md).

The synthetic-data generator provides a small software smoke input:

```bash
python scripts/generate_synthetic_data.py \
  --output .artifacts/synthetic \
  --height 64 \
  --width 96 \
  --timesteps 16 \
  --seed 2026
```

Files ending in `.sz3.out` and `.zfp.out` from this generator are deterministic corruption proxies. They are not produced by SZ3 or ZFP and cannot support a compressor-quality claim.

The matching smoke configuration is `configs/synthetic.yaml`. Its paths resolve to `.artifacts` at the repository root:

```bash
ptunet inspect-data --config configs/synthetic.yaml
ptunet train \
  --config configs/synthetic.yaml \
  --output .artifacts/smoke-run
ptunet evaluate \
  --config configs/synthetic.yaml \
  --checkpoint .artifacts/smoke-run/best.pt \
  --output .artifacts/smoke-evaluation
```

Choose fresh output directories when repeating the workflow.

## Validate configuration loading

Relative paths are resolved from the YAML file location. This small check prints the resolved roots without starting a run:

```bash
python - <<'PY'
from ptunet.config import load_config

config = load_config("path/to/experiment.yaml")
print(config.paths.data_root)
print(config.paths.output_root)
print([dataset.name for dataset in config.datasets])
PY
```

Loading rejects unknown keys, overlapping splits, invalid dimensions, unresolved environment variables, unknown compressor keys, and several incompatible model dimensions. File existence and exact byte count are checked when the dataset is opened.

The command-line equivalents are:

```bash
ptunet show-config --config path/to/experiment.yaml
ptunet model-summary --config path/to/experiment.yaml
ptunet inspect-data --config path/to/experiment.yaml
```

`show-config` prints a complete, reloadable YAML document with resolved absolute paths. Review those paths before sharing the output. `inspect-data` reads training targets to calculate normalization and checks the files needed to build every configured split. Training itself opens only the train and validation splits, so intentionally withheld test files do not block fitting.

## Train from the command line

```bash
ptunet train --config path/to/experiment.yaml
```

Use repeated `--set` arguments for recorded, typed overrides:

```bash
ptunet train \
  --config path/to/experiment.yaml \
  --set training.epochs=20 \
  --set training.device=cuda:0 \
  --output outputs/controlled-run
```

Resume a versioned package checkpoint with:

```bash
ptunet train \
  --config path/to/experiment.yaml \
  --resume path/to/last.pt
```

## Train through the Python API

```bash
python - <<'PY'
from ptunet.config import load_config
from ptunet.experiment import train_experiment

config = load_config("path/to/experiment.yaml")
run = train_experiment(config)
print(run.directory)
print(run.best_checkpoint)
PY
```

An automatically named run directory is created below `{output_root}/{experiment_name}` with a UTC timestamp. A requested output directory must be empty unless the workflow is resuming a checkpoint. This prevents an earlier run from being overwritten silently.

For an exact continuation, resume from `last.pt`:

```python
run = train_experiment(config, resume_checkpoint="path/to/last.pt")
```

Checkpoint format version 2 restores the model, optimizer, scheduler, precision scaler, best epoch and weights, early-stopping counter, complete history, random-number-generator state, and training-loader order. Format version 1 remains readable, but it cannot supply continuation fields that were not stored by that format.

Resume validates the model and stateful training settings recorded in the checkpoint. The epoch limit may stay unchanged or increase. Device, worker, and batch controls may change to fit the current host, but batch changes can alter optimization and should be reported. A new output directory receives rebuilt history records and a `resume.json` lineage file.

The Python API is useful in a notebook or a larger workflow. The command-line path is preferred for a recorded shell experiment because it keeps the invocation compact.

## Evaluate from the command line

```bash
ptunet evaluate \
  --config path/to/experiment.yaml \
  --checkpoint path/to/best.pt \
  --split test
```

## Evaluate a checkpoint through the Python API

```bash
python - <<'PY'
from ptunet.config import load_config
from ptunet.experiment import evaluate_experiment

config = load_config("path/to/experiment.yaml")
evaluation = evaluate_experiment(
    config,
    checkpoint="path/to/best.pt",
    split="test",
)
print(evaluation.metrics_csv)
PY
```

Accepted split names are `train`, `validation`, `val`, and `test`. Evaluation reconstructs each configured center field, writes the reconstructed float32 field, and records model and compressed-center baseline metrics. If no output directory is supplied, artifacts are written below the checkpoint directory in `evaluation/{split}`.

Evaluation requires saved normalization and model metadata. It rejects a behavior-changing model configuration even when tensor shapes happen to match. The destination must be new or empty so an earlier evaluation cannot be overwritten silently.

The Python function accepts `val` as an alias for `validation`. The command-line parser accepts `validation`.

## Run artifacts

A training run writes local, machine-readable records:

| File | Contents |
| --- | --- |
| `config.resolved.yaml` | Complete resolved experiment configuration |
| `environment.json` | Python, package, platform, Git, and accelerator metadata |
| `model.json` | Architecture values and parameter counts |
| `normalization.json` | Training-target normalization statistics |
| `metrics.jsonl` | Per-epoch local metric events |
| `summary.json` | Tracker summary and trainer settings |
| `history.csv` | Tabular training and validation history |
| `training_summary.json` | Best epoch, best validation loss, and stop status |
| `best.pt` | Best model checkpoint for evaluation and analysis |
| `last.pt` | Complete continuation checkpoint from the most recent epoch |

Evaluation writes `metrics.csv`, `evaluation.json`, and reconstructed binary fields. A metrics row records the resolved previous and next timesteps, compressor key, model metrics, compressed-center baseline metrics, inference time, patch count, and reconstruction path.

The artifact layout records what the software did. It does not establish data provenance or scientific validity by itself.

## Device and memory choices

`training.device: auto` uses CUDA when available and CPU otherwise. Use an explicit device for scheduled or multi-GPU systems. The package no longer sets `CUDA_VISIBLE_DEVICES` at import time. If a scheduler manages device visibility, set that environment variable in the job launcher and keep the YAML device relative to the visible set.

Full research configurations can create many overlapping patches. Adjust batch size and workers to the host, and begin with a short smoke run. Mixed precision is used only on CUDA when enabled.

## Tracking

Local JSON Lines tracking is always written for training. Weights & Biases is opt-in through configuration and the `tracking` optional dependency. Installing the dependency alone does not start a remote run.

```bash
python -m pip install -e '.[tracking]'
```

Do not store credentials in a configuration file or commit them to the repository.

## Checkpoint safety

Load checkpoints only from a trusted source. PTU-Net checkpoints contain PyTorch state and experiment metadata. The loader requests weights-only behavior when supported, but artifact provenance still matters. Do not treat an untrusted checkpoint as inert data.

## Migrating a prototype checkpoint

The model can translate parameter names from the format-version-2 wrapper written by the original script. Migration also calculates package normalization metadata from the configured training targets:

```bash
ptunet migrate-checkpoint \
  --config path/to/experiment.yaml \
  --legacy path/to/prototype.pth \
  --output path/to/migrated.pt
```

Use a model configuration that matches the prototype architecture. Migration does not prove numerical parity. Compare outputs on fixed inputs before using the converted checkpoint in an experiment.

`--allow-partial` permits missing or unexpected parameters and prints both lists. It can leave parameters at their current initialization, so it is a diagnostic option rather than evidence of a valid conversion.
