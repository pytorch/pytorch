# CUDA device guards for Python-native overrides. Python-native dispatch installs no device
# guard before an override runs, unlike the generated ATen path (#187983), so a kernel would
# otherwise launch on the current device's stream.

from __future__ import annotations

import functools
from typing import Any, TYPE_CHECKING, TypeVar

import torch


if TYPE_CHECKING:
    from collections.abc import Callable


R = TypeVar("R")


def cuda_device_guard(fn: Callable[..., R]) -> Callable[..., R]:
    """Run a CUDA override on its first tensor argument's device.

    Inputs already on the current device skip the context manager and its overhead.
    """

    @functools.wraps(fn)
    def guarded(self: torch.Tensor, *args: Any, **kwargs: Any) -> R:
        index = self.device.index
        if index == torch.cuda.current_device():
            return fn(self, *args, **kwargs)
        with torch.cuda.device(index):
            return fn(self, *args, **kwargs)

    return guarded
