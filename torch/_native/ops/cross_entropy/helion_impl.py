"""Helion override for pretuned B200 cross entropy shapes."""

from __future__ import annotations

import functools
from typing import Any

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
_REQUIRED_ALIGNMENT = 16

_autograd_lib: torch.library.Library | None = None
_autocast_lib: torch.library.Library | None = None


def _call_helion_read_only(kernel: Any, *args: object) -> torch.Tensor:
    from ...triton import ConstTensorWrapper

    # Wrap read-only launch args in ConstTensorWrapper so a copy-on-write input
    # is not materialized just to read it. Not ReadOnlyTensorWrapper: it is
    # DLPack-export-only, but a Triton launch duck-types data_ptr() off the arg.
    # Helion's host wrapper needs tensor metadata before the Triton launch, so
    # bind real tensors and use const-pointer wrappers only for the launch.
    raw_kernel = kernel.helion_kernel
    bound = raw_kernel.bind(args)
    bound.ensure_config_exists(args)
    launch_args = tuple(
        ConstTensorWrapper(arg) if isinstance(arg, torch.Tensor) else arg
        for arg in args
    )
    return bound(*launch_args)


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
    if torch.is_autocast_enabled("cuda"):
        return False
    # Target validation is data-dependent, so this override is eager-only.
    if type(self) is not torch.Tensor or type(target) is not torch.Tensor:
        return False
    self_ptr = self.const_data_ptr()  # pyrefly: ignore[missing-attribute]
    target_ptr = target.const_data_ptr()  # pyrefly: ignore[missing-attribute]
    if self_ptr % _REQUIRED_ALIGNMENT != 0 or target_ptr % _REQUIRED_ALIGNMENT != 0:
        return False

    from .helion_kernel import validate_labels

    with torch.accelerator.device_index(self.get_device()):
        if torch.cuda.is_current_stream_capturing():
            return False
        if torch._C._is_cow_tensor(target):  # pyrefly: ignore[missing-attribute]
            valid = _call_helion_read_only(validate_labels, target, self.shape[1])
        else:
            valid = validate_labels(target, self.shape[1])
        return bool(valid.item())


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
        is_cow = torch._C._is_cow_tensor  # pyrefly: ignore[missing-attribute]
        if is_cow(self) or is_cow(target):
            return _call_helion_read_only(cross_entropy, self, target)
        return cross_entropy(self, target)


def _autocast_cross_entropy(
    fallback_kernel: Any,
    keyset: torch._C.DispatchKeySet,
    self: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: int = 1,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    keyset = keyset.remove(torch._C.DispatchKey.AutocastCUDA)
    return fallback_kernel.call_boxed(
        keyset, self, target, weight, reduction, ignore_index, label_smoothing
    )


def _install_autograd_fallthrough(fallback_kernel: Any) -> None:
    # cross_entropy_loss is CompositeImplicitAutograd, so our CUDA router shadows
    # the composite kernel that otherwise gives it autograd and autocast (a plain
    # backend op needs neither -- see registry.py). Restore both: the AutogradCUDA
    # fallthrough lets backward decompose through aten (else PyTorch warns the op
    # has no autograd kernel), and the AutocastCUDA redispatch keeps
    # torch.compile + autocast in fp32 like aten.
    global _autocast_lib, _autograd_lib
    if _autograd_lib is not None:
        return
    op = "aten::cross_entropy_loss"
    has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key
    if has_kernel(op, "AutogradCUDA") or has_kernel(op, "AutocastCUDA"):
        return

    autocast_lib = torch.library.Library("aten", "IMPL", "AutocastCUDA")
    autocast_lib.impl(
        "cross_entropy_loss",
        functools.partial(_autocast_cross_entropy, fallback_kernel),
        with_keyset=True,
        allow_override=True,
    )
    try:
        autograd_lib = torch.library.Library("aten", "IMPL", "AutogradCUDA")
        autograd_lib.impl(
            "cross_entropy_loss", torch.library.fallthrough_kernel, allow_override=True
        )
    except Exception:
        autocast_lib._destroy()
        raise
    _autocast_lib = autocast_lib
    _autograd_lib = autograd_lib


def _uninstall_autograd_fallthrough() -> None:
    global _autocast_lib, _autograd_lib
    if _autograd_lib is None:
        return
    _autograd_lib._destroy()
    _autograd_lib = None
    if _autocast_lib is not None:
        _autocast_lib._destroy()
        _autocast_lib = None


def register_to_dispatch() -> None:
    hu.register_op_override(
        "aten",
        "cross_entropy_loss",
        "CUDA",
        cond=_cross_entropy_cond,
        impl=_cross_entropy_impl,
        allow_multiple_override=True,
    )
