# TL;DR This module is temporarily needed to avoid an unconditional import of numpy when importing torch.
# The functionality was factored out from `torch/testing/_internal/common_utils.py` because it is also needed in
# `torch/testing/_asserts.py`. The former unconditionally imports `numpy` while the latter is imported unconditionally
# in `torch/testing/__init__.py`. Since other modules import `torch.testing` we need this functionality in a module
# that does not depend on `numpy`.

import torch

__all__ = ["get_comparison_dtype", "DTYPE_PRECISIONS"]


# Acquires the comparison dtype, required since isclose
# requires both inputs have the same dtype, and isclose is not supported
# for some device x dtype combinations.
# NOTE: Remaps bfloat16 to float32 since neither the CPU or CUDA device types
#  support needed bfloat16 comparison methods.
# NOTE: Remaps float16 to float32 on CPU since the CPU device type doesn't
#   support needed float16 comparison methods.
# TODO: Update this once bfloat16 and float16 are better supported.
def get_comparison_dtype(a, b):
    # TODO: update this when promote_types supports bfloat16 and/or
    # isclose supports bfloat16.
    a_dtype = torch.float32 if a.dtype is torch.bfloat16 else a.dtype
    b_dtype = torch.float32 if b.dtype is torch.bfloat16 else b.dtype

    compare_dtype = torch.promote_types(a_dtype, b_dtype)

    # non-CUDA (CPU, for example) float16 -> float32
    # TODO: update this when isclose is implemented for CPU float16
    if (compare_dtype is torch.float16 and
        (a.device != b.device or a.device.type != 'cuda' or
            b.device.type != 'cuda')):
        compare_dtype = torch.float32

    return compare_dtype


# Some analysis of tolerance by logging tests from test_torch.py can be found
# in https://github.com/pytorch/pytorch/pull/32538.
# dtype name : (rtol, atol)
DTYPE_PRECISIONS = {
    torch.float16    : (0.001, 1e-5),
    torch.bfloat16   : (0.016, 1e-5),
    torch.float32    : (1.3e-6, 1e-5),
    torch.float64    : (1e-7, 1e-7),
    torch.complex32  : (0.001, 1e-5),
    torch.complex64  : (1.3e-6, 1e-5),
    torch.complex128 : (1e-7, 1e-7),
}
