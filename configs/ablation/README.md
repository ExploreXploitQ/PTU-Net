# Ablation configurations

These small configurations use the deterministic synthetic dataset. They test
which prediction stage contributes to reconstruction error:

| Configuration | Adaptive baseline | Transformer correction | U-Net refinement |
| --- | --- | --- | --- |
| `baseline-only.yaml` | yes | no | no |
| `no-transformer.yaml` | yes | no | yes |
| `no-unet.yaml` | yes | yes | no |
| `../synthetic.yaml` | yes | yes | yes |

Generate the data once, then train each configuration with the same seed. The
repository does not include reference results. Record generated run directories
and compare their `history.csv` and test `metrics.csv` files.
