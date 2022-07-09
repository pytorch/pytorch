import torch

from torch._prims.utils import (
    check,
    is_low_precision_dtype,
    is_float_dtype,
    is_complex_dtype,
)


def check_fp_or_complex(
    dtype: torch.dtype, fn_name: str, allow_low_precision_dtypes: bool = True
):
    """
    Checks whether the input is floating point or complex.
    If allow_low_precision_dtypes is True, it allows having float16, bfloat16, and complex32
    """
    check(
        is_float_dtype(dtype) or is_complex_dtype(dtype),
        lambda: f"{fn_name}: Expected a floating point or complex tensor as input. Got {dtype}",
    )
    check(
        allow_low_precision_dtypes or not is_low_precision_dtype(dtype),
        lambda: f"{fn_name}: Half precision dtypes not supported. Got {dtype}",
    )
