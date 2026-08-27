PYTHON ?= python3

.PHONY: install install-dev test test-cov lint format typecheck check smoke clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e '.[dev,tracking]'

test:
	$(PYTHON) -m pytest

test-cov:
	$(PYTHON) -m pytest --cov=ptunet --cov-report=term-missing --cov-report=xml

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

typecheck:
	$(PYTHON) -m mypy src/ptunet

check: lint typecheck test

smoke:
	$(PYTHON) scripts/generate_synthetic_data.py --output .artifacts/synthetic
	ptunet train --config configs/synthetic.yaml

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; paths = [Path(p) for p in ['.coverage', 'coverage.xml', 'htmlcov', '.pytest_cache', '.ruff_cache', '.mypy_cache', 'build', 'dist', '.artifacts']] + [p for root in [Path('src'), Path('tests'), Path('scripts')] for p in root.rglob('__pycache__')] + list(Path('src').glob('*.egg-info')); [shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True) for p in paths]"
