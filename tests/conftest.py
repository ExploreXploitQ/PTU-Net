from __future__ import annotations

from typing import Any

import torch


def pytest_sessionstart(session: Any) -> None:
    del session
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
