# Reproducibility

## Current evidence boundary

The repository packages an existing research implementation, but it does not distribute the data or trained weights used by the original experiment. No benchmark values have been accepted as repository results. A clean installation and unit tests can check software behavior, but they cannot reproduce an empirical result without the original inputs and a recorded experiment manifest.

## Information required for a reproducible run

Archive the following with every reported experiment:

- the Git commit and any local patch;
- the complete resolved configuration;
- Python, PyTorch, CUDA, cuDNN, GPU, driver, and operating system versions;
- random seeds and deterministic-algorithm settings;
- file identities or checksums, dimensions, preprocessing, and split membership;
- temporal offsets and the compressor selected for each input timestep;
- checkpoint selection rule and saved epoch;
- metric definitions, aggregation rules, and data range;
- wall-clock timing boundaries, warmup policy, and synchronization policy.

Keep raw logs and machine-readable metrics. A table copied into Markdown is not sufficient evidence by itself.

## Legacy script assumptions

The original `ptu-net_001_1.py` file contains several assumptions that must not silently carry into a portable experiment:

- `CUDA_VISIBLE_DEVICES` is set to device index 3 at import time.
- The input data root is an absolute path for one host.
- Weights & Biases starts whenever the optional package imports successfully.
- Training and validation datasets calculate normalization statistics independently in the legacy loader.
- The saved inference normalization values come from the training dataset.
- Test boundary offsets are chosen by evaluation-list position, while another helper chooses offsets by file availability.
- The legacy best-state assignment uses a shallow state-dictionary copy. The package stores detached best weights and complete continuation state in versioned checkpoints.
- GPU availability selects the device automatically, without an explicit device argument.

The package validation loop applies the same composite objective used for training when it schedules learning rates, selects a checkpoint, and applies early stopping. The prototype selected on validation MSE alone. This is a recorded behavioral difference, so numerical parity should not be claimed without an explicit comparison.

The package refactor is intended to make these choices explicit. Until parity tests cover a behavior, document whether a run follows the legacy behavior or the refactored behavior.

## Determinism

Setting a seed is necessary but not sufficient for bitwise reproducibility. PyTorch operations, data-loader workers, CUDA kernels, library versions, and GPU architecture can change numerical results. Report the determinism mode used and distinguish exact reproducibility from statistical repeatability.

Use a small synthetic dataset for smoke tests. Synthetic success verifies file handling, tensor shapes, optimization flow, checkpoint round trips, and metric plumbing. It does not validate scientific performance.

Checkpoint-based evaluation requires the saved model configuration and normalization metadata. Resume additionally compares stateful optimizer, scheduler, loss, and warmup settings, then restores random-number-generator and loader state. These checks prevent a command-line override from silently changing the meaning of a continued run.

## Comparison rules

Compare methods only when they use the same target fields, temporal centers, compressor outputs, normalization source, and metric implementation. Report absolute reconstruction metrics together with the compressed center-field baseline. Include variability across repeated runs when training randomness can affect the conclusion.

See [results.md](results.md) for the minimum record required before a number is presented as a project result.
