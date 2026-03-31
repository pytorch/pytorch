import functools

import torch
from torch import Tensor

from ... import cutedsl_utils as cu


def _should_use_oink_softmax(input: object, dim: int) -> bool:
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
    if dim != -1 and dim != input.ndim - 1:
        return False
    if input.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


def _softmax_out_impl(
    dispatch_keys,
    self,
    dim,
    half_to_float,
    out,
    *,
    _fallback_kernel=None,
):
    if not half_to_float and _should_use_oink_softmax(self, dim):
        from .softmax_kernels import softmax_forward

        result = softmax_forward(self)
        out.copy_(result)
        return out
    return _fallback_kernel.call_boxed(dispatch_keys, self, dim, half_to_float, out)


def _register_oink_softmax() -> None:
    if not cu.runtime_available():
        return
    fallback = torch.library.get_kernel("aten::_softmax.out", "CUDA")
    cu.register_op_override(
        "aten",
        "_softmax.out",
        "CUDA",
        functools.partial(_softmax_out_impl, _fallback_kernel=fallback),
    )


_register_oink_softmax()
