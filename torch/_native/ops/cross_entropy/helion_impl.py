"""Helion override for pretuned B200 cross entropy shapes."""

from __future__ import annotations

import torch

from ... import helion_utils as hu


_B200_PRETUNED_SHAPES: frozenset[tuple[int, int]] = frozenset(
    {
        (2048, 32000),
        (4096, 32000),
        (8192, 32000),
        (8192, 128000),
        (16384, 128000),
        (32768, 128000),
        (2048, 128256),
        (4096, 128256),
        (8192, 128256),
        (16384, 128256),
        (2048, 129280),
        (4096, 129280),
        (8192, 129280),
        (2048, 151936),
        (4096, 151936),
        (8192, 151936),
        (2048, 152064),
        (4096, 152064),
        (8192, 152064),
        (1024, 256000),
        (2048, 256000),
    }
)

# The target-value eligibility check requires a device-to-host synchronization.
# At this smallest shape, that fixed cost makes the integrated override slower
# than ATen on B200; every other pretuned shape wins in end-to-end measurement.
_B200_SHAPES = _B200_PRETUNED_SHAPES - {(2048, 32000)}

_autograd_lib: torch.library.Library | None = None


def _cross_entropy_cond(
    self: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: int = 1,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> bool:
    if self.device.type != "cuda" or torch.version.hip is not None:
        return False
    if torch.cuda.get_device_capability(self.device) != (10, 0):
        return False
    if self.dtype != torch.bfloat16 or target.dtype != torch.int64:
        return False
    if self.ndim != 2 or target.ndim != 1:
        return False
    if (self.shape[0], self.shape[1]) not in _B200_SHAPES:
        return False
    if target.shape[0] != self.shape[0] or target.device != self.device:
        return False
    if not self.is_contiguous() or not target.is_contiguous():
        return False
    if weight is not None or reduction != 1 or ignore_index != -100:
        return False
    if label_smoothing != 0.0:
        return False
    if self.requires_grad or target.requires_grad:
        return False
    if torch.autograd.forward_ad.unpack_dual(self).tangent is not None:
        return False
    is_cow = torch._C._is_cow_tensor  # pyrefly: ignore[missing-attribute]
    if is_cow(self) or is_cow(target):
        return False

    from .helion_kernel import validate_labels

    with torch.accelerator.device_index(self.get_device()):
        if torch.cuda.is_current_stream_capturing():
            return False
        return bool(validate_labels(target, self.shape[1]).item())


def _cross_entropy_impl(
    self: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: int = 1,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    from .helion_kernel import cross_entropy

    with torch.accelerator.device_index(self.get_device()):
        return cross_entropy(self, target)


def _install_autograd_fallthrough() -> None:
    global _autograd_lib
    if _autograd_lib is not None:
        return
    _autograd_lib = torch.library.Library("aten", "IMPL", "AutogradCUDA")
    _autograd_lib.impl(
        "cross_entropy_loss", torch.library.fallthrough_kernel, allow_override=True
    )


def _uninstall_autograd_fallthrough() -> None:
    global _autograd_lib
    if _autograd_lib is None:
        return
    _autograd_lib._destroy()
    _autograd_lib = None


def register_to_dispatch() -> None:
    if (
        not hu.runtime_available()
        or not hu._version_is_sufficient()
        or hu.check_native_jit_disabled()
    ):
        return
    hu.register_op_override(
        "aten",
        "cross_entropy_loss",
        "CUDA",
        cond=_cross_entropy_cond,
        impl=_cross_entropy_impl,
        allow_multiple_override=True,
    )
