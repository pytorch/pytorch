"""Export torch work functions for binary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `torch._numpy/_ufuncs.py` module.
"""

import torch

from torch import (  # noqa: F401
    add,  # noqa: F401
    arctan2,  # noqa: F401
    bitwise_and,  # noqa: F401
    bitwise_left_shift as left_shift,  # noqa: F401
    bitwise_or,  # noqa: F401
    bitwise_right_shift as right_shift,  # noqa: F401
    bitwise_xor,  # noqa: F401
    copysign,  # noqa: F401
    divide,  # noqa: F401
    eq as equal,  # noqa: F401
    float_power,  # noqa: F401
    floor_divide,  # noqa: F401
    fmax,  # noqa: F401
    fmin,  # noqa: F401
    fmod,  # noqa: F401
    gcd,  # noqa: F401
    greater,  # noqa: F401
    greater_equal,  # noqa: F401
    heaviside,  # noqa: F401
    hypot,  # noqa: F401
    lcm,  # noqa: F401
    ldexp,  # noqa: F401
    less,  # noqa: F401
    less_equal,  # noqa: F401
    logaddexp,  # noqa: F401
    logaddexp2,  # noqa: F401
    logical_and,  # noqa: F401
    logical_or,  # noqa: F401
    logical_xor,  # noqa: F401
    maximum,  # noqa: F401
    minimum,  # noqa: F401
    multiply,  # noqa: F401
    nextafter,  # noqa: F401
    not_equal,  # noqa: F401
    pow as power,  # noqa: F401
    remainder,  # noqa: F401
    remainder as mod,  # noqa: F401
    subtract,  # noqa: F401
    true_divide,  # noqa: F401
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
    is_half = dtype == torch.float16

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


# a stub implementation of divmod, should be improved after
# https://github.com/pytorch/pytorch/issues/90820 is fixed in pytorch
def divmod(x, y):
    return x // y, x % y
