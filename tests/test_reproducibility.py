from __future__ import annotations

import numpy as np
import torch

from ptunet.reproducibility import collect_environment, seed_everything


def test_seed_everything_repeats_cpu_streams() -> None:
    seed_everything(17)
    first_numpy = np.random.random(4)
    first_torch = torch.rand(4)
    seed_everything(17)

    np.testing.assert_array_equal(np.random.random(4), first_numpy)
    torch.testing.assert_close(torch.rand(4), first_torch)


def test_environment_record_has_core_versions(tmp_path) -> None:
    record = collect_environment(tmp_path)

    assert record["python"]
    assert record["packages"]["torch"]
    assert isinstance(record["cuda_devices"], list)
