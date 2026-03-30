import functools

import torch
from torch import Tensor

from ... import cutedsl_utils as cu


def _should_use_oink_rmsnorm(input: object) -> bool:
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


def _fused_rms_norm_impl(
    dispatch_keys,
    input,
    normalized_shape,
    weight,
    eps,
    *,
    _fallback_kernel=None,
):
    if _should_use_oink_rmsnorm(input):
        from .rmsnorm_kernels import rmsnorm_forward

        y, rstd, _ = rmsnorm_forward(
            input, weight, eps=eps if eps is not None else 1e-5, store_rstd=True
        )
        return (y, rstd)
    return _fallback_kernel.call_boxed(
        dispatch_keys, input, normalized_shape, weight, eps
    )


def _register_oink_rmsnorm() -> None:
    fallback = torch.library.get_kernel("aten::_fused_rms_norm", "CUDA")
    cu.register_op_override(
        "aten",
        "_fused_rms_norm",
        "CUDA",
        functools.partial(_fused_rms_norm_impl, _fallback_kernel=fallback),
    )


_register_oink_rmsnorm()
