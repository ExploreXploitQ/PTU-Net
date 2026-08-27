"""Binary field I/O and compressor-aware path resolution."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ptunet.config import DatasetSpec

FLOAT32_DTYPE = np.dtype("<f4")


class FieldIOError(ValueError):
    """Raised when a field file or configured file layout is invalid."""


def _shape_tuple(shape: int | Sequence[int]) -> tuple[int, ...]:
    result = (shape,) if isinstance(shape, int) else tuple(shape)
    invalid_size = any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in result
    )
    if not result or invalid_size:
        raise FieldIOError(f"shape must contain positive integers, got {shape!r}")
    return result


def validate_float32_file(path: str | os.PathLike[str], shape: int | Sequence[int]) -> Path:
    """Validate that ``path`` is exactly one little-endian float32 field."""

    field_path = Path(path)
    dimensions = _shape_tuple(shape)
    expected_bytes = int(np.prod(dimensions, dtype=np.int64)) * FLOAT32_DTYPE.itemsize
    try:
        actual_bytes = field_path.stat().st_size
    except OSError as error:
        raise FieldIOError(f"cannot stat field file {field_path}: {error}") from error
    if not field_path.is_file():
        raise FieldIOError(f"field path is not a regular file: {field_path}")
    if actual_bytes != expected_bytes:
        raise FieldIOError(
            f"field {field_path} has {actual_bytes} bytes; expected exactly "
            f"{expected_bytes} for shape {dimensions} and dtype {FLOAT32_DTYPE.str}"
        )
    return field_path


def open_float32_memmap(
    path: str | os.PathLike[str],
    shape: int | Sequence[int],
    *,
    mode: Literal["r", "r+", "c"] = "r",
) -> np.memmap:
    """Open an exact-size raw field lazily as a little-endian float32 memmap."""

    if mode not in {"r", "r+", "c"}:
        raise FieldIOError("existing field memmaps support only modes 'r', 'r+', and 'c'")
    dimensions = _shape_tuple(shape)
    field_path = validate_float32_file(path, dimensions)
    return np.memmap(field_path, dtype=FLOAT32_DTYPE, mode=mode, shape=dimensions, order="C")


@dataclass(frozen=True)
class CompressorResolver:
    """Resolve parity-dependent compressors and their on-disk suffixes."""

    extensions: Mapping[str, str]

    def __post_init__(self) -> None:
        extensions = dict(self.extensions)
        if not extensions:
            raise FieldIOError("at least one compressor extension is required")
        for compressor, extension in extensions.items():
            if not isinstance(compressor, str) or not compressor:
                raise FieldIOError("compressor names must be non-empty strings")
            if not isinstance(extension, str):
                raise FieldIOError(f"extension for {compressor!r} must be a string")
            if "/" in extension or (os.altsep and os.altsep in extension):
                raise FieldIOError(f"extension for {compressor!r} must be a suffix, not a path")
        object.__setattr__(self, "extensions", extensions)

    def compressor_for(self, spec: DatasetSpec, timestep: int) -> str:
        compressor = spec.compressor_for_timestep(timestep)
        self.extension_for(compressor)
        return compressor

    def extension_for(self, compressor: str) -> str:
        try:
            return self.extensions[compressor]
        except KeyError as error:
            available = ", ".join(sorted(self.extensions))
            raise FieldIOError(
                f"unknown compressor {compressor!r}; configured compressors: {available}"
            ) from error


def _template_values(
    spec: DatasetSpec,
    timestep: int,
    *,
    compressor: str = "",
    extension: str = "",
) -> dict[str, Any]:
    if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 0:
        raise FieldIOError(f"timestep must be a non-negative integer, got {timestep!r}")
    return {
        "name": spec.name,
        "variable": spec.variable,
        "resolution_tag": spec.resolution_tag,
        "height": spec.height,
        "width": spec.width,
        "timestep": f"{timestep:0{spec.timestep_digits}d}",
        "timestep_index": timestep,
        "compressor": compressor,
        "extension": extension,
    }


def _render(template: str, values: Mapping[str, Any], label: str) -> Path:
    try:
        rendered = template.format_map(values)
    except (KeyError, ValueError) as error:
        raise FieldIOError(f"invalid {label} template {template!r}: {error}") from error
    path = Path(rendered)
    if path.is_absolute():
        raise FieldIOError(f"{label} template must render a relative path, got {path}")
    return path


def _under(root: str | os.PathLike[str], relative: Path, label: str) -> Path:
    root_path = Path(root).expanduser().resolve()
    candidate = (root_path / relative).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as error:
        raise FieldIOError(f"{label} path escapes configured root: {relative}") from error
    return candidate


def original_path(spec: DatasetSpec, data_root: str | os.PathLike[str], timestep: int) -> Path:
    values = _template_values(spec, timestep)
    relative = _render(spec.original_template, values, "original")
    return _under(data_root, relative, "original")


def compressed_path(
    spec: DatasetSpec,
    data_root: str | os.PathLike[str],
    timestep: int,
    resolver: CompressorResolver,
    *,
    compressor: str | None = None,
) -> Path:
    selected = compressor or resolver.compressor_for(spec, timestep)
    extension = resolver.extension_for(selected)
    values = _template_values(spec, timestep, compressor=selected, extension=extension)
    relative = _render(spec.compressed_template, values, "compressed")
    return _under(data_root, relative, "compressed")


def reconstruction_path(
    spec: DatasetSpec,
    output_root: str | os.PathLike[str],
    timestep: int,
    resolver: CompressorResolver,
    *,
    compressor: str | None = None,
) -> Path:
    selected = compressor or resolver.compressor_for(spec, timestep)
    extension = resolver.extension_for(selected)
    values = _template_values(spec, timestep, compressor=selected, extension=extension)
    subdirectory = _render(spec.output_subdir, values, "output_subdir")
    filename = _render(spec.reconstruction_template, values, "reconstruction")
    return _under(output_root, subdirectory / filename, "reconstruction")


def write_float32_atomic(
    path: str | os.PathLike[str], values: np.ndarray, *, create_parents: bool = True
) -> Path:
    """Write a C-order little-endian float32 array and atomically replace ``path``."""

    destination = Path(path)
    if create_parents:
        destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=FLOAT32_DTYPE, order="C")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            array.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise FieldIOError(f"cannot write field {destination}: {error}") from error
    return destination


__all__ = [
    "FLOAT32_DTYPE",
    "CompressorResolver",
    "FieldIOError",
    "compressed_path",
    "open_float32_memmap",
    "original_path",
    "reconstruction_path",
    "validate_float32_file",
    "write_float32_atomic",
]
