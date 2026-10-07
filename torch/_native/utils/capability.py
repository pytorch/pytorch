# Cheap, non-raising per-tensor capability gates run on every eager dispatch. Op families
# add their own dtype and geometry checks.

from __future__ import annotations

import functools
from typing import Any, TYPE_CHECKING, TypeVar

import torch
import torch._subclasses.fake_tensor


if TYPE_CHECKING:
    from collections.abc import Callable


R = TypeVar("R")


def is_traced(t: torch.Tensor) -> bool:
    # Fake and meta tensors have no launchable storage. The exact-type check is faster than
    # is_fake() (0.52 vs 1.01us), but C++ dispatch-key wrappers can also be exact tensors.
    if type(t) is torch.Tensor:
        return (
            t.device.type == "meta"
            or torch._is_functional_tensor(t)
            or torch._C._is_fake_tensor(t)
        )
    return torch._subclasses.fake_tensor.is_fake(t) or t.device.type == "meta"


@functools.cache
def _arch_ok(idx: int, majors: tuple[int, ...]) -> bool:
    # Device capability is immutable; memoize it with the caller's accepted set in the key.
    try:
        major, _ = torch.cuda.get_device_capability(idx)
    except RuntimeError:
        # Fake CUDA tensors can reach here on CPU-only systems, where the query raises.
        # Capability conditions must decline rather than propagate the error.
        return False
    return major in majors


def device_ok(x: torch.Tensor, majors: tuple[int, ...]) -> bool:
    """Whether x is CUDA, not HIP, on a caller-supported compute-capability major.
    Majors are an explicit allowlist because families differ; this avoids admitting
    untested hardware such as SM 11.0.
    """
    if x.device.type != "cuda" or torch.version.hip is not None:
        return False
    return _arch_ok(x.device.index, tuple(majors))


def on_current_device(x: torch.Tensor) -> bool:
    if x.device.type != "cuda":
        return False
    try:
        return x.device.index == torch.cuda.current_device()
    except RuntimeError:
        return False


def cuda_device_guard(fn: Callable[..., R]) -> Callable[..., R]:
    """Run a CUDA override on its first tensor argument's device."""

    @functools.wraps(fn)
    def guarded(self: torch.Tensor, *args: Any, **kwargs: Any) -> R:
        index = self.device.index
        if index == torch.cuda.current_device():
            return fn(self, *args, **kwargs)
        with torch.cuda.device(index):
            return fn(self, *args, **kwargs)

    return guarded


def resolve_neg_view(fn: Callable[..., R]) -> Callable[..., R]:
    """Materialize a lazy negative input before passing it to an override."""

    @functools.wraps(fn)
    def resolved(self: torch.Tensor, *args: Any, **kwargs: Any) -> R:
        if self.is_neg():
            self = self.resolve_neg()
        return fn(self, *args, **kwargs)

    return resolved
