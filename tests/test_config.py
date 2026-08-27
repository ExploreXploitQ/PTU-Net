from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ptunet.config import ConfigError, DatasetSpec, load_config


def _write_config(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_resolves_paths_and_nested_sections(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PTUNET_TEST_OUTPUT", str(tmp_path / "job-output"))
    path = _write_config(
        tmp_path / "configs",
        """
name: portable
paths:
  data_root: ../data
  output_root: ${PTUNET_TEST_OUTPUT}
datasets:
  - name: pressure
    variable: pressure
    resolution_tag: 4x5
    shape: [4, 5]
    splits:
      train: {start: 1, end: 5, step: 2}
      val: [6]
      test: [7]
    compressors: {even: sz3, odd: hpez}
model:
  patch_size: 4
  stride: 2
  subpatch_size: 2
training:
  batch_size: 16
tracking:
  enabled: true
  tags: [paper, ablation]
""",
    )

    config = load_config(
        path,
        overrides=["training.epochs=12", "evaluation.temporal_search_radius=3"],
    )

    assert config.name == "portable"
    assert config.paths.data_root == (tmp_path / "data").resolve()
    assert config.paths.output_root == (tmp_path / "job-output").resolve()
    assert config.datasets[0].train_timesteps == (1, 3, 5)
    assert config.datasets[0].validation_timesteps == (6,)
    assert config.datasets[0].compressor_for_timestep(6) == "sz3"
    assert config.datasets[0].compressor_for_timestep(7) == "hpez"
    assert config.model.subpatch_size == 2
    assert config.model.mlp_ratio == 2.0
    assert config.model.correction_scale_init == 0.02
    assert config.training.epochs == 12
    assert config.training.batch_size == 16
    assert config.training.learning_rate == 5.0e-4
    assert config.evaluation.temporal_search_radius == 3
    assert config.tracking.tags == ("paper", "ablation")


def test_mapping_override_preserves_other_training_values(tmp_path) -> None:
    path = _write_config(
        tmp_path,
        """
paths: {data_root: data}
datasets:
  - name: field
    variable: v
    resolution_tag: tiny
    shape: [8, 8]
    splits: {train: [1]}
training:
  epochs: 20
  batch_size: 7
  loss: {mse: 0.6, charbonnier: 0.4, correction: 0.002}
""",
    )

    config = load_config(path, overrides={"training": {"epochs": 3, "loss": {"mse": 0.8}}})

    assert config.training.epochs == 3
    assert config.training.batch_size == 7
    assert config.training.loss.mse == 0.8
    assert config.training.loss.charbonnier == 0.4
    assert config.training.loss.correction == 0.002
    assert config.paths.output_root == (tmp_path / "outputs").resolve()


def test_unknown_keys_and_unknown_compressors_are_rejected(tmp_path) -> None:
    unknown_key = _write_config(
        tmp_path / "unknown-key",
        """
paths: {data_root: data}
datasets:
  - name: field
    variable: v
    resolution_tag: tiny
    shape: [8, 8]
    splits: {train: [1]}
training: {epochz: 2}
""",
    )
    with pytest.raises(ConfigError, match="epochz"):
        load_config(unknown_key)

    unknown_compressor = _write_config(
        tmp_path / "unknown-compressor",
        """
paths: {data_root: data}
datasets:
  - name: field
    variable: v
    resolution_tag: tiny
    shape: [8, 8]
    splits: {train: [1]}
    compressors: {even: imaginary, odd: imaginary}
""",
    )
    with pytest.raises(ConfigError, match="unknown compressor"):
        load_config(unknown_compressor)


def test_dataset_spec_rejects_split_leakage() -> None:
    with pytest.raises(ConfigError, match="overlapping train/validation"):
        DatasetSpec(
            name="field",
            variable="v",
            resolution_tag="tiny",
            height=4,
            width=4,
            train_timesteps=(1, 2),
            validation_timesteps=(2, 3),
        )


def test_resolved_config_round_trip_uses_canonical_schema(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PTUNET_TEST_OUTPUT", str(tmp_path / "job-output"))
    source_path = _write_config(
        tmp_path / "source",
        """
name: round-trip
paths:
  data_root: ../scientific-data
  output_root: ${PTUNET_TEST_OUTPUT}
compressors:
  extensions:
    quant: .quant.bin
datasets:
  - name: pressure
    variable: pres
    resolution_tag: 6x8
    shape: [6, 8]
    timestep_digits: 3
    splits:
      train: {start: 1, end: 5, step: 2}
      validation: [6]
      test: [8]
    compressors: {even: quant, odd: sz3}
    output_subdir: recon/{name}/{compressor}
    templates:
      original: raw-{timestep}.bin
      compressed: compressed-{timestep}{extension}
      reconstruction: reconstructed-{timestep}.bin
model:
  patch_size: 4
  stride: 2
  subpatch_size: 2
  baseline_prior: [0.1, 0.8, 0.1]
training:
  epochs: 2
  loss: {mse: 0.8, charbonnier: 0.2, correction: 0.001}
tracking:
  tags: [round-trip, resolved]
""",
    )

    original = load_config(source_path)
    document = original.to_dict()

    assert "compressor_extensions" not in document
    assert document["compressors"]["extensions"]["quant"] == ".quant.bin"
    assert document["datasets"][0]["shape"] == [6, 8]
    assert document["datasets"][0]["splits"] == {
        "train": [1, 3, 5],
        "validation": [6],
        "test": [8],
    }
    assert document["datasets"][0]["compressors"] == {"even": "quant", "odd": "sz3"}
    assert document["model"]["baseline_prior"] == [0.1, 0.8, 0.1]
    assert document["tracking"]["tags"] == ["round-trip", "resolved"]
    assert document["paths"] == {
        "data_root": str((tmp_path / "scientific-data").resolve()),
        "output_root": str((tmp_path / "job-output").resolve()),
    }

    # This follows the same dump path as ``ptunet show-config`` and experiment
    # run manifests.  Moving the resolved YAML must not rebase its paths.
    rendered = yaml.safe_dump(document, sort_keys=False)
    assert "!!python" not in rendered
    resolved_path = _write_config(tmp_path / "moved", rendered)
    monkeypatch.delenv("PTUNET_TEST_OUTPUT")

    reloaded = load_config(resolved_path)

    assert reloaded == original
    assert reloaded.paths.data_root == (tmp_path / "scientific-data").resolve()
    assert reloaded.paths.output_root == (tmp_path / "job-output").resolve()
