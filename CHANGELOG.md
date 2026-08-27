# Changelog

This file records user-visible changes. The project has not yet published a stable release.

## Unreleased

### Added

- Initial package structure for the PTU-Net research implementation.
- Configuration, data loading, model, training, evaluation, and command-line modules.
- Unit tests and a synthetic-data path for local checks.
- Public documentation and repository contribution guidance.
- Reloadable resolved configurations and matched-seed component ablations.
- Atomic checkpoint format version 2 with full resume state and version-1 loading compatibility.
- Checkpoint model-configuration validation and checkpoint-owned normalization for evaluation and resume.
- Lazy training that does not require test files, plus CPU and optional CUDA test paths.

### Status

- No trained checkpoint, research dataset, benchmark table, or verified performance result is distributed with the repository.
- Version `0.1.0` in package metadata identifies the current alpha code line. It does not by itself indicate a published release.
