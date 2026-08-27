#!/usr/bin/env python3
"""Generate a small temporal field dataset for PTU-Net smoke experiments.

The generated ``.sz3.out`` and ``.zfp.out`` files are deterministic corruption
proxies. They are not outputs from the named compressor implementations and
must not be used for compressor-quality claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FLOAT32_LE = np.dtype("<f4")


def analytic_field(timestep: int, height: int, width: int) -> np.ndarray:
    """Create a smooth field with moving waves and a compact rotating feature."""

    y, x = np.mgrid[0:height, 0:width]
    x = x.astype(np.float64) / max(width - 1, 1)
    y = y.astype(np.float64) / max(height - 1, 1)
    phase = timestep / 12.0
    waves = np.sin(2.0 * np.pi * (2.0 * x + 0.35 * phase))
    waves += 0.55 * np.cos(2.0 * np.pi * (y - 0.2 * phase))
    center_x = 0.5 + 0.18 * np.cos(2.0 * np.pi * phase)
    center_y = 0.5 + 0.18 * np.sin(2.0 * np.pi * phase)
    radius_squared = (x - center_x) ** 2 + (y - center_y) ** 2
    compact = 1.4 * np.exp(-radius_squared / 0.012)
    trend = 0.15 * phase * (x - y)
    return (waves + compact + trend).astype(np.float32)


def corruption_proxy(
    field: np.ndarray,
    timestep: int,
    compressor: str,
    seed: int,
) -> np.ndarray:
    """Apply deterministic quantization and low-amplitude structured error."""

    profiles = {
        "sz3": (0.025, 0.006),
        "zfp": (0.040, 0.004),
    }
    quantization, noise_scale = profiles[compressor]
    rng = np.random.default_rng(seed + 1009 * timestep + (0 if compressor == "sz3" else 1))
    quantized = np.round(field / quantization) * quantization
    noise = rng.normal(0.0, noise_scale, size=field.shape)
    return (quantized + noise).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination data directory")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height < 8 or args.width < 8 or args.timesteps < 5:
        raise SystemExit("height and width must be at least 8; timesteps must be at least 5")
    args.output.mkdir(parents=True, exist_ok=True)
    resolution = f"{args.height}x{args.width}"
    for timestep in range(args.timesteps):
        field = analytic_field(timestep, args.height, args.width)
        stem = f"SYNTH_{resolution}_{timestep:04d}.dat"
        np.asarray(field, dtype=FLOAT32_LE).tofile(args.output / stem)
        for compressor, extension in (("sz3", ".sz3.out"), ("zfp", ".zfp.out")):
            reconstruction = corruption_proxy(field, timestep, compressor, args.seed)
            np.asarray(reconstruction, dtype=FLOAT32_LE).tofile(args.output / f"{stem}{extension}")
    print(
        f"wrote {args.timesteps} original fields and two corruption proxies per field "
        f"to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
