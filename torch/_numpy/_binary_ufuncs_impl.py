# mypy: ignore-errors

"""Export torch work functions for binary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `torch._numpy/_ufuncs.py` module.
"""

import torch
from torch import (  # noqa: F401
    add,
    arctan2,
    bitwise_and,
    bitwise_left_shift as left_shift,
    bitwise_or,
    bitwise_right_shift as right_shift,
    bitwise_xor,
    copysign,
    divide,
    eq as equal,
    float_power,
    floor_divide,
    fmax,
    fmin,
    fmod,
    gcd,
    greater,
    greater_equal,
    heaviside,
    hypot,
    lcm,
    ldexp,
    less,
    less_equal,
    logaddexp,
    logaddexp2,
    logical_and,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    multiply,
    nextafter,
    not_equal,
    pow as power,
    remainder,
    remainder as mod,
    true_divide,
)

from . import _dtypes_impl, _util


# work around torch limitations w.r.t. numpy
def matmul(x, y):
    # work around:
    #  - RuntimeError: expected scalar type Int but found Double
    #  - RuntimeError: "addmm_impl_cpu_" not implemented for 'Bool'
    #  - RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    dtype = _dtypes_impl.result_type_impl(x, y)
    is_bool = dtype == torch.bool
    is_half = (x.dtype == torch.float16 or y.dtype == torch.float16) and (
        x.is_cpu or y.is_cpu
    )

    work_dtype = dtype
    if is_bool:
        work_dtype = torch.uint8
    if is_half:
        work_dtype = torch.float32

    x = _util.cast_if_needed(x, work_dtype)
    y = _util.cast_if_needed(y, work_dtype)

    result = torch.matmul(x, y)

    if work_dtype != dtype:
        result = result.to(dtype)

    return result


def _is_bool(x):
    return _dtypes_impl._dtype_for_scalar_or_tensor(x) == torch.bool


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=_dtypes_impl._dtype_for_scalar_or_tensor(x))


def subtract(x, y):
    if _is_bool(x) or _is_bool(y):
        x = _as_tensor(x)
        y = _as_tensor(y)
        dtype = _dtypes_impl.result_type_impl(x, y)
        if dtype == torch.bool:
            raise TypeError(
                "numpy boolean subtract, the `-` operator, is not supported, "
                "use the bitwise_xor, the `^` operator, or the logical_xor "
                "function instead."
            )
        x = _util.cast_if_needed(x, dtype)
        y = _util.cast_if_needed(y, dtype)
    return torch.subtract(x, y)


# a stub implementation of divmod, should be improved after
# https://github.com/pytorch/pytorch/issues/90820 is fixed in pytorch
def divmod(x, y):
    return x // y, x % y
