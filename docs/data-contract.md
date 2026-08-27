# Data contract

The package reads flat, headerless binary fields with NumPy. A dataset configuration supplies the dimensions and filename components needed to interpret each file.

## Array representation

Each field must contain exactly `height * width` little-endian `float32` values in row-major order. The package represents this dtype as `<f4` and validates the byte count before opening a memory map. The original script used native-endian `numpy.float32`, so data produced on a big-endian system needs explicit conversion for the package.

Headerless binary files do not encode shape, byte order, units, missing-value conventions, or variable metadata. Record these properties outside the file and validate them before training. Converting data to a self-describing format is recommended for long-term exchange, but that conversion is not part of the current implementation.

## File naming

An original field uses:

```text
{variable}_{resolution_tag}_{timestep}.dat
```

The timestep is zero-padded to the configured width. A compressed input appends a compressor-specific suffix to the original filename:

| Compressor key | Suffix |
| --- | --- |
| `sz3` | `.sz3.out` |
| `sz` | `.sz.out` |
| `szp` | `.szp.out` |
| `sperr` | `.sperr.out` |
| `hpez` | `.hpez.out` |
| `zfp` | `.zfp.out` |
| `mgard` | `.mgard.out` |
| `fpzip` | `.fpzip.out` |

For example, a compressed field can be named:

```text
CLDHGH_1800x3600_07.dat.zfp.out
```

The code reads already reconstructed compressor output. It does not invoke a compressor or decompress an encoded bitstream. Output reconstruction names are also configurable. The package default retains the compressor suffix before `.recon.dat` so outputs from different compressor assignments can remain distinct.

## Temporal samples

A standard interior sample uses three compressed fields centered on timestep `t` and the original field at `t` as its target:

```text
input:  compressed(t - 1), compressed(t), compressed(t + 1)
target: original(t)
```

The compressor resolver can assign one compressor to even timesteps and another to odd timesteps. Parity is evaluated on the integer timestep itself.

The package resolves temporal windows from compressed files that exist within a configured search radius. It prefers the nearest available timestep on each side. At a lower boundary with two later fields it uses `(t + 2, t, t + 1)`. At an upper boundary with two earlier fields it uses `(t - 1, t, t - 2)`. If only one neighboring field exists, the missing side repeats the center and contributes a zero difference channel. The center remains the semantic target in every case.

The original script contains two boundary-selection paths that are not identical. Treat temporal offset selection as part of the experiment definition and record the resolved window for every evaluated center.

## Normalization

The package calculates one population mean and standard deviation from original training target fields across all configured datasets. The calculation is streaming and does not concatenate all fields in memory. The same saved statistics must be used for training inputs, validation, test evaluation, and checkpoint-based inference.

Normalization metadata also records the training-target count, minimum, and maximum. It rejects non-finite values and zero-variance training data. This package behavior differs from the original script, where the training and validation dataset objects calculated their own statistics.

## Historical experiment declaration

The original script declares four variables named `CLDHGH`, `CLDLOW`, `CLDMED`, and `CLDTOT`, each with shape 1800 by 3600. It declares training centers 1 through 9, validation centers 12 and 13, and test centers 15 through 62. The active compressor pairing in the checked script is `sz3` on even timesteps and `zfp` on odd timesteps.

These declarations document code state only. The corresponding data is not distributed here, the split has not been audited for a published benchmark, and no result is claimed for it.

## Validation checklist

Before a run, verify:

1. every requested original and compressed file exists;
2. each file contains exactly the configured number of values;
3. dimensions, byte order, units, and missing values are correct;
4. temporal neighbors do not cross an invalid sequence boundary;
5. train, validation, and test centers are disjoint as intended;
6. the compressor assignment matches the files on disk;
7. the data root and output root point to separate, writable locations.
