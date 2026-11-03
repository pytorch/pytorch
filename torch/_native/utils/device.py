# CUDA device guards for Python-native overrides. Python-native dispatch installs no device
# guard before an override runs, unlike the generated C++ ATen path, so a kernel would
# otherwise launch on the current device's stream (#187983 fixed the same bug for bmm).
# TODO: topk/cutedsl_impl.py and topk/flydsl_impl.py still inline their own copies.

from __future__ import annotations

import functools
from typing import Concatenate, ParamSpec, TYPE_CHECKING, TypeVar

import torch


if TYPE_CHECKING:
    from collections.abc import Callable


P = ParamSpec("P")
R = TypeVar("R")


def cuda_device_guard(
    fn: Callable[Concatenate[torch.Tensor, P], R],
) -> Callable[Concatenate[torch.Tensor, P], R]:
    """Run an override on its first positional argument's CUDA device.

    Inputs already on the current device, and non-CUDA inputs, skip the context manager
    and its overhead.
    """

    @functools.wraps(fn)
    def guarded(self: torch.Tensor, *args: P.args, **kwargs: P.kwargs) -> R:
        # get_device() is -1 off CUDA and avoids building a torch.device.
        index = self.get_device()
        if index < 0 or index == torch.cuda.current_device():
            return fn(self, *args, **kwargs)
        with torch.cuda.device(index):
            return fn(self, *args, **kwargs)

    return guarded
