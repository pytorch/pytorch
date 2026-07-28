"""Overrides for aten::native_dropout on CUDA.

Both DSL implementations reproduce aten's fused_dropout_kernel_vec exactly
(same philox stream, same launch math, same generator offset advancement) and
draw RNG state through ``Generator.philox_state``. Both are registered
(cutedsl first, so it wins when both are enabled); select one with the
torch.backends.python_native per-DSL controls.
"""

import torch

from ... import cutedsl_utils, triton_utils
from ._common import eligible


def _cond(x: torch.Tensor, p: float, train=None, *args, **kwargs) -> bool:
    try:
        return eligible(x, p, train)
    except Exception:
        return False


def _cutedsl_impl(x, p, train=None):
    from .cutedsl_kernels import dropout_fwd

    return dropout_fwd(x, p)


def _triton_impl(x, p, train=None):
    from .triton_kernels import dropout_fwd

    return dropout_fwd(x, p)


def register_to_dispatch() -> None:
    if not torch.backends.cuda.is_built():
        return
    cutedsl_utils.register_op_override(
        "aten", "native_dropout", "CUDA", cond=_cond, impl=_cutedsl_impl
    )
    triton_utils.register_op_override(
        "aten",
        "native_dropout",
        "CUDA",
        cond=_cond,
        impl=_triton_impl,
        allow_multiple_override=True,
    )
