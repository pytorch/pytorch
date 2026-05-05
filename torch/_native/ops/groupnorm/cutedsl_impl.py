"""CuTeDSL override for aten::native_group_norm{,_backward}."""
# mypy: allow-untyped-defs

from __future__ import annotations

import functools

import torch

from ... import cutedsl_utils as cu


@functools.cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


@functools.cache
def _get_groupnorm_kernels():
    from .norms import cutedsl_groupnorm_bwd, cutedsl_groupnorm_fwd

    return cutedsl_groupnorm_fwd, cutedsl_groupnorm_bwd


def _cutedsl_native_group_norm_impl(
    dispatch_keys: torch.DispatchKeySet,
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    eps: float,
    *,
    fallback_kernel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    K = (C // group) * HxW
    use_fallback = (
        _get_device_major(input.device) not in (9, 10)
        or input.dtype not in _supported_dtypes
        or K < 32
        or K > 1024 * 1024
    )
    if use_fallback:
        return fallback_kernel.call_boxed(
            dispatch_keys, input, weight, bias, N, C, HxW, group, eps
        )

    cutedsl_groupnorm_fwd, _ = _get_groupnorm_kernels()
    return cutedsl_groupnorm_fwd(input, weight, bias, N, C, HxW, group, eps)


def _cutedsl_native_group_norm_backward_impl(
    dispatch_keys: torch.DispatchKeySet,
    grad_out: torch.Tensor,
    input: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    output_mask: list[bool],
    *,
    fallback_kernel,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    _supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    K = (C // group) * HxW
    use_fallback = (
        _get_device_major(input.device) not in (9, 10)
        or input.dtype not in _supported_dtypes
        or K < 32
        or K > 1024 * 1024
    )
    if use_fallback:
        return fallback_kernel.call_boxed(
            dispatch_keys,
            grad_out,
            input,
            mean,
            rstd,
            weight,
            N,
            C,
            HxW,
            group,
            output_mask,
        )

    _, cutedsl_groupnorm_bwd = _get_groupnorm_kernels()
    return cutedsl_groupnorm_bwd(
        grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
    )


def _register_for_dispatch_key(dispatch_key: str) -> None:
    fwd_fallback = torch.library.get_kernel("aten::native_group_norm", dispatch_key)
    bwd_fallback = torch.library.get_kernel(
        "aten::native_group_norm_backward", dispatch_key
    )

    cu.register_op_override(
        "aten",
        "native_group_norm",
        dispatch_key,
        functools.partial(
            _cutedsl_native_group_norm_impl, fallback_kernel=fwd_fallback
        ),
        allow_multiple_override=True,
    )
    cu.register_op_override(
        "aten",
        "native_group_norm_backward",
        dispatch_key,
        functools.partial(
            _cutedsl_native_group_norm_backward_impl, fallback_kernel=bwd_fallback
        ),
        allow_multiple_override=True,
    )


def register_to_dispatch() -> None:
    if not cu.runtime_available():
        return

    _register_for_dispatch_key("CUDA")
