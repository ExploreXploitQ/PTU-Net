<p align="center">
  <img src="assets/ptu-net-wordmark.svg" width="760" alt="PTU-Net, Patch Transformer U-Net research software">
</p>

<p align="center">
  <a href="https://github.com/ExploreXploitQ/PTU-Net/actions/workflows/ci.yml"><img src="https://github.com/ExploreXploitQ/PTU-Net/actions/workflows/ci.yml/badge.svg" alt="Continuous integration status"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/status-alpha-C47F17" alt="Project status: alpha">
</p>

# PTU-Net

PTU-Net is an alpha research package for reconstructing a two-dimensional center field from three temporally adjacent compressed fields. The model combines a center-anchored temporal baseline, a subpatch transformer correction, and a gated U-Net refinement head.

> **Evidence status:** The architecture and research workflow are implemented in this repository. No research dataset, trained checkpoint, benchmark table, or verified performance result is included. Accuracy, runtime, memory use, and generalization remain unevaluated at the repository level.

## What the model does

Given compressed fields from the previous, center, and next timesteps, PTU-Net forms three channels:

```text
center
previous - center
next - center
```

It then builds an adaptive temporal baseline, predicts a bounded correction with a transformer, and applies a gated convolutional refinement. Full-field inference combines overlapping patches with Gaussian weights.

![Architecture diagram showing temporal inputs, center anchoring, adaptive baseline, transformer correction, gated U-Net, and reconstruction](assets/architecture.svg)

The original experiment uses 32 by 32 patches, stride 16, 8 by 8 subpatches, embedding width 384, 12 attention heads, and 8 transformer layers. These values are defaults inherited from the implementation, not an empirical recommendation.

## Research workflow

- Validated YAML records paths, field layouts, temporal splits, compressor routing, model choices, optimization, evaluation, and optional tracking.
- Training fields are opened lazily with bounded memory-map caching. Normalization is fitted once on training targets and reused without validation or test leakage.
- Atomic checkpoint format version 2 preserves the optimizer, scheduler, precision scaler, best model, early-stopping state, history, random-number generators, and data-loader order.
- Evaluation checks checkpoint architecture metadata before loading, reports the compressed center field as a baseline, and writes per-field metrics with reconstructed float32 outputs.
- Explicit pathway switches and matched-seed configurations support baseline-only, transformer-removal, and U-Net-removal studies.

## Status at a glance

| Item | Status |
| --- | --- |
| Package maturity | Alpha research software |
| Core model | Implemented |
| Configuration and command line | Implemented in the package refactor |
| Atomic checkpoint and exact resume state | Implemented, with format-version-1 loading compatibility |
| Synthetic software checks | Included for behavior and workflow testing |
| Research data | Not distributed |
| Trained weights | Not distributed |
| Published benchmark values | None in this repository |
| Legacy parity | Requires further equivalence testing |
| Software license | Not declared |

## Installation

PTU-Net requires Python 3.10 or newer. Install it from source in a dedicated environment:

```bash
git clone https://github.com/ExploreXploitQ/PTU-Net.git
cd PTU-Net
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On a CUDA server, install a PyTorch build compatible with that server before installing PTU-Net. The package uses PyTorch's visible devices and does not bundle a CUDA runtime or choose a global GPU index.

For development checks:

```bash
python -m pip install -e '.[dev]'
make check
```

Weights & Biases support is optional:

```bash
python -m pip install -e '.[tracking]'
```

## Synthetic workflow check

The repository includes a small deterministic workflow for checking installation, data loading, training, checkpointing, and evaluation:

```bash
python scripts/generate_synthetic_data.py --output .artifacts/synthetic
ptunet inspect-data --config configs/synthetic.yaml
ptunet train \
  --config configs/synthetic.yaml \
  --output .artifacts/smoke-run
ptunet evaluate \
  --config configs/synthetic.yaml \
  --checkpoint .artifacts/smoke-run/best.pt \
  --output .artifacts/smoke-evaluation
```

Explicit output directories must be new or empty. Choose a different run name when repeating the check.

The files labeled `sz3` and `zfp` in this synthetic workflow are corruption proxies created by the repository script. They are not compressor outputs. A successful smoke run is a software check, not evidence of reconstruction quality or compressor performance.

## Small model check

This example checks the model interface on CPU. The input is already in center-difference form and does not represent a scientific evaluation.

```python
import torch

from ptunet.models import PTUNet, PTUNetConfig

config = PTUNetConfig(
    patch_size=16,
    subpatch_size=8,
    embed_dim=32,
    num_heads=4,
    num_layers=1,
    baseline_hidden_channels=8,
    unet_base_channels=8,
)
model = PTUNet(config).eval()
inputs = torch.randn(2, 3, 16, 16)

with torch.no_grad():
    reconstruction = model(inputs)

assert reconstruction.shape == (2, 16, 16)
```

The installed command is available as either:

```bash
ptunet --help
python -m ptunet --help
```

See [the usage guide](docs/usage.md) before preparing a training or evaluation run. It explains configuration resolution, data validation, synthetic smoke tests, checkpoints, and output files.

## Data boundary

The package reads little-endian float32 fields with an exact configured shape. It expects already reconstructed compressor output, not encoded compressor bitstreams. Filename templates and parity-based compressor selection are explicit configuration values.

The original script names four cloud-related variables, but the associated data source, provenance, units, and usage terms are not included. Those names do not establish a public dataset or a benchmark.

Read [the data contract](docs/data-contract.md) before using local data.

## Documentation

- [Architecture](docs/architecture.md)
- [Data contract](docs/data-contract.md)
- [Usage](docs/usage.md)
- [Configuration](docs/configuration.md)
- [Reproducibility](docs/reproducibility.md)
- [Model card](docs/model-card.md)
- [Results and evaluation status](docs/results.md)

## Repository layout

```text
src/ptunet/     Python package
configs/        Example experiment configurations
scripts/        Synthetic data and repository utilities
tests/          Unit and workflow tests
docs/           Technical and research documentation
assets/         Repository-native SVG figures
```

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and evidence requirements. Report suspected vulnerabilities according to [SECURITY.md](SECURITY.md). Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Citation

Use the metadata in [CITATION.cff](CITATION.cff) and cite the exact version or commit used. The repository does not claim an associated paper or institutional affiliation.

## License status

This repository does not currently contain a license grant. Standard copyright restrictions therefore apply. Citation does not grant permission to copy, modify, or redistribute the software.
