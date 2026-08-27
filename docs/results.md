# Results and evaluation status

## Current status

No benchmark result is published as a PTU-Net project result at this time.

| Artifact | Repository status |
| --- | --- |
| Research dataset | Not distributed |
| Dataset provenance and usage terms | Not documented |
| Trained checkpoint | Not distributed |
| Raw experiment log | Not distributed |
| Machine-readable benchmark output | Not distributed |
| Verified accuracy or quality metric | Not reported |
| Verified runtime or memory result | Not reported |

This table is an evidence statement, not an indication that the architecture has failed or succeeded. It prevents implementation details and locally observed output from being presented as validated findings.

## Metrics implemented by the original script

For reconstruction \(r\) and target \(y\), mean squared error is the mean of \((r-y)^2\). The package PSNR implementation uses an explicit positive data range when one is configured. Otherwise, it uses the reference maximum minus its minimum. The original script instead used the maximum absolute target value. Record which definition produced a value and do not compare the two as if they were identical.

The package function named `global_ssim` calculates one statistic from global means, variances, and covariance. It does not calculate the usual local-window SSIM map. Publications and comparisons should call it a global SSIM-like statistic unless equivalence to a specific standard is demonstrated.

The original evaluation baseline MSE compares the target with the compressed center field. Its reported PSNR improvement is

\[
10 \log_{10}\left(\frac{\mathrm{MSE}_{baseline}}{\mathrm{MSE}_{model}}\right).
\]

Reconstruction time in the original evaluation covers patch dataset construction, model execution, host transfer, and patch blending after the input arrays have been loaded. It does not include reading input files or writing the reconstructed field. Hardware, warmup, synchronization, and repeated-trial policy must accompany any timing result.

## Requirements for publishing a result

A pull request that adds a result should include:

1. a precise question or hypothesis;
2. dataset provenance, usage terms, checksums, preprocessing, dimensions, and splits;
3. the Git commit and resolved configuration;
4. environment and hardware details;
5. random seeds and repeated-run policy;
6. baseline definitions and metric implementations;
7. machine-readable per-sample output;
8. a script that regenerates the summary from that output;
9. limitations and known sources of uncertainty.

Label exploratory runs as preliminary. Do not select only favorable samples or runs. When data cannot be redistributed, provide a documented acquisition route and enough metadata for an authorized user to verify identity.
