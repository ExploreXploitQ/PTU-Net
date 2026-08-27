# Model card

## Model details

PTU-Net is a patch-based neural reconstruction model for two-dimensional scientific fields. It combines a learned temporal baseline, a transformer correction path, and a gated U-Net refinement head. Package metadata currently identifies the code line as version `0.1.0` with alpha development status.

No trained weights are distributed with the repository.

## Intended use

The implementation is intended for research on reconstructing an uncompressed center field from three already reconstructed lossy-compression outputs sampled at nearby timesteps. It can also support software tests with synthetic float32 fields.

Any use with a new variable, grid, temporal cadence, compressor, or error tolerance requires a new validation. The code should not be treated as a drop-in correction method without that study.

## Out-of-scope use

The repository is not designed or validated for safety-critical decisions, operational forecasting, medical data, autonomous control, or recovering information that was never present in the inputs. It does not provide guarantees about physical conservation, uncertainty, worst-case error, or behavior outside the training distribution.

## Inputs and outputs

The logical input is a previous, center, and next compressed field with a common two-dimensional grid. The model converts them to a normalized center channel and two center-relative temporal differences. The output is one reconstructed center field with the configured height and width.

The legacy data path expects native-endian float32 binary arrays. See [data-contract.md](data-contract.md) for naming and shape requirements.

## Training objective

The original training loop combines normalized mean squared error, Charbonnier loss, and an L2 penalty on the transformer correction map. Its default weights are 0.7, 0.3, and 0.0001 respectively. The two learned global baseline parameters use a five-epoch warmup, while the spatial baseline gate follows the normal parameter groups. AdamW, gradient clipping, a plateau scheduler, and validation-based early stopping are used.

These settings describe implementation defaults. They are not established optimal settings.

## Evaluation status

The code defines MSE, PSNR, a global SSIM-like statistic, compressed-center baseline MSE, PSNR improvement, and reconstruction time. The repository currently reports no accepted values for these metrics. The global statistic is not a windowed SSIM implementation and should be named precisely in comparisons.

## Limitations

- Empirical accuracy, runtime, memory use, and generalization have not been established in this repository.
- Headerless binary input depends on external shape and metadata.
- The temporal formulation assumes aligned grids and meaningful neighboring timesteps.
- Patch processing can introduce boundary or blending effects.
- A point estimate does not express reconstruction uncertainty.
- The model does not impose physical constraints or conservation laws.
- Compressor-specific behavior can become part of the learned distribution.
- The original script includes host-specific device and path settings.

## Data and environmental considerations

The original experiment names four cloud-related variables but does not include their source, provenance, collection protocol, units, or usage terms. Do not infer dataset permission or scientific representativeness from variable names. Training cost and energy use depend on the chosen grid, patch count, model size, epoch count, and hardware and have not been measured here.

## Citation and license status

Citation metadata is provided in [`CITATION.cff`](../CITATION.cff). The repository does not currently declare a software license. Citation does not grant reuse or redistribution rights.
