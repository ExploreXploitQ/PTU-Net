# Contributing

PTU-Net is an early research software project. Contributions that improve correctness, reproducibility, documentation, or test coverage are welcome.

## Before opening a change

Search the issue tracker for related work. For changes to data formats, model behavior, evaluation definitions, or public APIs, open an issue before writing a large patch. Describe the scientific or engineering reason for the change and how it can be checked.

This repository does not currently declare a software license. Do not assume permission to redistribute or reuse the code. If licensing affects a proposed contribution, discuss it with the maintainers first.

## Development setup

Use Python 3.10 or newer in a dedicated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Run the local checks before opening a pull request:

```bash
make check
```

The equivalent commands are:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src/ptunet
python -m pytest -m 'not gpu' --cov=ptunet --cov-report=term-missing
```

## Scientific changes

Keep implementation facts separate from empirical findings.

- Include a small deterministic test when changing tensor shapes, patch placement, normalization, checkpoint handling, or metric calculations.
- Record the data split, seed, package versions, device, and command for any reported result.
- Compare metrics on the same samples and with the same data range.
- Do not add a performance claim without the underlying machine-readable output and a reproducible command.
- Mark exploratory numbers as preliminary. Do not present them as project results.

Large datasets, generated reconstructions, experiment logs, and checkpoints should remain outside Git. A small synthetic fixture is acceptable when it is needed for a test.

## Pull requests

Keep each pull request focused. Explain the observed behavior, the proposed change, and how you tested it. Update public documentation when configuration fields, file layouts, commands, or output formats change.

By participating, you agree to follow [the code of conduct](CODE_OF_CONDUCT.md).
