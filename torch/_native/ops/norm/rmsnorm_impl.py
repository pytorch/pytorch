"""Quack-backed RMSNorm overrides for aten fused RMSNorm operators.

Requires the `quack-kernels` package (https://github.com/Dao-AILab/quack)
When quack is not installed the overrides are silently skipped
"""
# mypy: allow-untyped-defs

from __future__ import annotations

import functools
import importlib
import logging
import os
from collections.abc import Callable

import torch
from torch._vendor.packaging.version import Version

from ... import cutedsl_utils as cu
from ...common_utils import _available_version, check_native_version_skip


log = logging.getLogger(__name__)


_QUACK_REQUIRED_VERSIONS: set[Version] = {
    Version(f"{0}.{3}.{7}"),
}


def _quack_available() -> bool:
    # Disable quack's .o disk cache before first import — loading
    # cached objects can segfault due to a quack jit_cache bug.
    # Must be set before find_spec because that triggers quack.__init__
    # which imports quack.cache_utils.
    os.environ.setdefault("QUACK_CACHE_ENABLED", "0")
    if importlib.util.find_spec("quack") is None:
        return False
    return True


@functools.cache
def _quack_version_is_ok() -> bool:
    version = _available_version("quack-kernels")
    if check_native_version_skip() or (version in _QUACK_REQUIRED_VERSIONS):
        return True

    log.info(
        "quack-kernels version %s is not known-good (ok: %s); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        _QUACK_REQUIRED_VERSIONS,
    )
    return False


_RMSNormFwdFallback = Callable[
    [torch.DispatchKeySet, torch.Tensor, list[int], torch.Tensor | None, float | None],
    tuple[torch.Tensor, torch.Tensor],
]
_RMSNormBwdFallback = Callable[
    [
        torch.DispatchKeySet,
        torch.Tensor,
        torch.Tensor,
        list[int],
        torch.Tensor,
        torch.Tensor | None,
        list[bool],
    ],
    tuple[torch.Tensor | None, torch.Tensor | None],
]


@functools.cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


def _is_supported(input: torch.Tensor) -> bool:
    return input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ) and _get_device_major(input.device) in (9, 10)


def _fused_rms_norm_impl(
    dispatch_keys: torch.DispatchKeySet,
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    eps: float | None,
    *,
    fallback_kernel: _RMSNormFwdFallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_supported(input):
        return fallback_kernel.call_boxed(  # pyrefly: ignore[missing-attribute]
            dispatch_keys, input, normalized_shape, weight, eps
        )

    if eps is None:
        eps = torch.finfo(input.dtype).eps

    from .norms import quack_rmsnorm_fwd

    return quack_rmsnorm_fwd(input, weight, normalized_shape, eps)


def _fused_rms_norm_backward_impl(
    dispatch_keys: torch.DispatchKeySet,
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: list[int],
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask: list[bool],
    *,
    fallback_kernel: _RMSNormBwdFallback,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not _is_supported(input):
        return fallback_kernel.call_boxed(  # pyrefly: ignore[missing-attribute]
            dispatch_keys,
            grad_out,
            input,
            normalized_shape,
            rstd,
            weight,
            output_mask,
        )

    from .norms import quack_rmsnorm_bwd

    grad_input, grad_weight = quack_rmsnorm_bwd(
        grad_out,
        input,
        rstd,
        weight,
        normalized_shape,
        dw_mask=output_mask[1],
    )

    if not output_mask[0]:
        grad_input = None
    return grad_input, grad_weight


def register_rmsnorm_overrides() -> None:
    if not _quack_available():
        log.debug("quack-kernels not installed, skipping RMSNorm overrides")
        return

    if not _quack_version_is_ok():
        return

    if not torch.cuda.is_available():
        return

    fwd_fallback = torch.library.get_kernel("aten::_fused_rms_norm", "CUDA")
    bwd_fallback = torch.library.get_kernel("aten::_fused_rms_norm_backward", "CUDA")

    fwd_impl = functools.partial(
        _fused_rms_norm_impl,
        fallback_kernel=fwd_fallback,
    )
    bwd_impl = functools.partial(
        _fused_rms_norm_backward_impl,
        fallback_kernel=bwd_fallback,
    )

    cu.register_op_override("aten", "_fused_rms_norm", "CUDA", fwd_impl)
    cu.register_op_override("aten", "_fused_rms_norm_backward", "CUDA", bwd_impl)


register_rmsnorm_overrides()
