import functools

import torch
from torch import Tensor

from ... import cutedsl_utils as cu


def _should_use_oink_layernorm(input: object) -> bool:
    if not cu.runtime_available():
        return False
    if not isinstance(input, Tensor):
        return False
    if input.device.type != "cuda":
        return False
    try:
        major, _ = torch.cuda.get_device_capability(input.device)
    except Exception:
        return False
    if major < 10:
        return False
    if input.ndim != 2:
        return False
    if input.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


def _native_layer_norm_impl(
    dispatch_keys,
    input,
    normalized_shape,
    weight,
    bias,
    eps,
    *,
    _fallback_kernel=None,
):
    if _should_use_oink_layernorm(input):
        from .layernorm_kernels import layernorm

        # oink layernorm returns (out, rstd, mean) with both flags True
        # aten native_layer_norm returns (out, mean, rstd)
        out, rstd, mean = layernorm(
            input, weight, bias=bias, eps=eps, return_rstd=True, return_mean=True
        )
        return (out, mean, rstd)
    return _fallback_kernel.call_boxed(
        dispatch_keys, input, normalized_shape, weight, bias, eps
    )


def _register_oink_layernorm() -> None:
    if not cu.runtime_available():
        return
    fallback = torch.library.get_kernel("aten::native_layer_norm", "CUDA")
    cu.register_op_override(
        "aten",
        "native_layer_norm",
        "CUDA",
        functools.partial(_native_layer_norm_impl, _fallback_kernel=fallback),
    )


_register_oink_layernorm()
