"""Helion override for pretuned B200 cross entropy shapes."""

from __future__ import annotations

import functools
import os
import threading
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

# The validation and launch overhead makes the smallest pretuned shape slower
# than ATen on B200; every other pretuned shape wins end-to-end.
_B200_SHAPES = _B200_PRETUNED_SHAPES - {(2048, 32000)}
_REQUIRED_ALIGNMENT = 16

_autograd_lib: torch.library.Library | None = None
_autocast_lib: torch.library.Library | None = None
_registration_lock = threading.RLock()


def _reset_registration_lock_after_fork() -> None:
    global _registration_lock
    _registration_lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_registration_lock_after_fork)


@functools.cache
def _is_sm100(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (10, 0)


def _call_helion_read_only(kernel: Any, *args: object) -> Any:
    from ...triton import ConstTensorWrapper

    def launch() -> Any:
        # Helion needs tensor metadata while binding; const-pointer wrappers are
        # only used for the launch so read-only COW inputs stay unmaterialized.
        bound = kernel.helion_kernel.bind(args)
        bound.ensure_config_exists(args)
        launch_args = tuple(
            ConstTensorWrapper(arg)
            if isinstance(arg, torch.Tensor)
            and torch._C._is_cow_tensor(arg)  # pyrefly: ignore[missing-attribute]
            else arg
            for arg in args
        )
        return bound(*launch_args)

    return kernel.run_with_instrumentation(launch, *args)


def _cross_entropy_cond(
    self: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: int = 1,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> bool:
    # Tensor subclasses use ATen; this shape-specialized override is eager-only.
    if type(self) is not torch.Tensor or type(target) is not torch.Tensor:
        return False
    if self.shape not in _B200_SHAPES:
        return False
    if self.device.type != "cuda" or torch.version.hip is not None:
        return False
    if self.dtype != torch.bfloat16 or target.dtype != torch.int64:
        return False
    if self.ndim != 2 or target.ndim != 1:
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
    self_ptr = self.const_data_ptr()  # pyrefly: ignore[missing-attribute]
    target_ptr = target.const_data_ptr()  # pyrefly: ignore[missing-attribute]
    if self_ptr % _REQUIRED_ALIGNMENT != 0 or target_ptr % _REQUIRED_ALIGNMENT != 0:
        return False
    if not _is_sm100(self.get_device()):
        return False
    with torch.accelerator.device_index(self.get_device()):
        return not torch.cuda.is_current_stream_capturing()


def _cross_entropy_impl(
    self: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: int = 1,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    from .helion_kernel import cross_entropy, validate_labels_and_count

    with torch.accelerator.device_index(self.get_device()):
        is_cow = torch._C._is_cow_tensor  # pyrefly: ignore[missing-attribute]
        if is_cow(target):
            valid, nonignored_count = _call_helion_read_only(
                validate_labels_and_count, target, self.shape[1]
            )
        else:
            valid, nonignored_count = validate_labels_and_count(target, self.shape[1])
        torch._assert_async(valid, "Target is out of bounds.")
        if is_cow(self) or is_cow(target):
            return _call_helion_read_only(cross_entropy, self, target, nonignored_count)
        return cross_entropy(self, target, nonignored_count)


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
    # FakeTensor dispatch during Dynamo tracing must use the captured kernel to
    # avoid re-entering this AutocastCUDA registration through the aten op.
    if torch.compiler.is_dynamo_compiling():
        return fallback_kernel.call_boxed(
            keyset, self, target, weight, reduction, ignore_index, label_smoothing
        )
    if type(self) is not torch.Tensor or type(target) is not torch.Tensor:
        from torch._subclasses.fake_tensor import is_fake

        if is_fake(self) or is_fake(target):
            return fallback_kernel.call_boxed(
                keyset, self, target, weight, reduction, ignore_index, label_smoothing
            )
    return torch.ops.aten.cross_entropy_loss.default.redispatch(
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
    with _registration_lock:
        op = "aten::cross_entropy_loss"
        has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key
        new_autocast_lib = None
        new_autograd_lib = None
        try:
            if _autocast_lib is None and not has_kernel(op, "AutocastCUDA"):
                new_autocast_lib = torch.library.Library("aten", "IMPL", "AutocastCUDA")
                _autocast_lib = new_autocast_lib
                new_autocast_lib.impl(
                    "cross_entropy_loss",
                    functools.partial(_autocast_cross_entropy, fallback_kernel),
                    with_keyset=True,
                    allow_override=True,
                )
            if _autograd_lib is None and not has_kernel(op, "AutogradCUDA"):
                new_autograd_lib = torch.library.Library("aten", "IMPL", "AutogradCUDA")
                _autograd_lib = new_autograd_lib
                new_autograd_lib.impl(
                    "cross_entropy_loss",
                    torch.library.fallthrough_kernel,
                    allow_override=True,
                )
        except BaseException:
            if new_autograd_lib is not None:
                new_autograd_lib._destroy()
                _autograd_lib = None
            if new_autocast_lib is not None:
                new_autocast_lib._destroy()
                _autocast_lib = None
            raise


def _uninstall_autograd_fallthrough() -> None:
    global _autocast_lib, _autograd_lib
    with _registration_lock:
        if _autograd_lib is not None:
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
