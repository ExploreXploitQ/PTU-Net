# Documentation

PTU-Net is research software for reconstructing a two-dimensional center field from three temporally adjacent compressed fields. The repository separates the reusable package from the original host-specific experiment script.

## Project status

The package is in alpha development. No research dataset, trained checkpoint, benchmark table, or independently verified metric is included. The architecture is implemented, but empirical quality and performance claims remain to be established with reproducible experiments.

The repository does not currently declare a software license. Review the repository status before reuse or redistribution.

## Guides

- [Architecture](architecture.md) explains the model data flow and tensor conventions.
- [Data contract](data-contract.md) defines binary field naming, shapes, temporal inputs, and compressor routing.
- [Usage](usage.md) covers installation and command-line workflows.
- [Configuration](configuration.md) documents the experiment schema and overrides.
- [Reproducibility](reproducibility.md) records the current evidence boundary and known migration risks.
- [Model card](model-card.md) states intended uses and limitations.
- [Results](results.md) defines which metrics exist and why no benchmark values are reported yet.

For changes to scientific behavior or public interfaces, see [the contribution guide](../CONTRIBUTING.md).
