"""
The testing package contains testing-specific utilities.
"""

import torch
import random
import math
import cmath
from typing import cast, List, Optional, Tuple, Union
from .check_kernel_launches import check_cuda_kernel_launches, check_code_for_cuda_kernel_launches
import operator

FileCheck = torch._C.FileCheck

__all__ = [
    'assert_allclose', 'make_non_contiguous', 'rand_like', 'randn_like'
]

rand_like = torch.rand_like
randn_like = torch.randn_like

# Helper function that returns True when the dtype is an integral dtype,
# False otherwise.
# TODO: implement numpy-like issubdtype
def is_integral(dtype: torch.dtype) -> bool:
    # Skip complex/quantized types
    dtypes = [x for x in get_all_dtypes() if x not in get_all_complex_dtypes()]
    return dtype in dtypes and not dtype.is_floating_point

def is_quantized(dtype: torch.dtype) -> bool:
    return dtype in (torch.quint8, torch.qint8, torch.qint32, torch.quint4x2)

# Helper function that maps a flattened index back into the given shape
# TODO: consider adding torch.unravel_index
def _unravel_index(flat_index, shape):
    flat_index = operator.index(flat_index)
    res = []

    # Short-circuits on zero dim tensors
    if shape == torch.Size([]):
        return 0

    for size in shape[::-1]:
        res.append(flat_index % size)
        flat_index = flat_index // size

    if len(res) == 1:
        return res[0]

    return tuple(res[::-1])
# (bool, msg) tuple, where msg is None if and only if bool is True.
_compare_return_type = Tuple[bool, Optional[str]]

# Compares two tensors with the same size on the same device and with the same
# dtype for equality.
# Returns a tuple (bool, msg). The bool value returned is True when the tensors
# are "equal" and False otherwise.
# The msg value is a debug string, and is None if the tensors are "equal."
# NOTE: Test Framework Tensor 'Equality'
#   Two tensors are "equal" if they are "close", in the sense of torch.allclose.
#   The only exceptions are complex tensors and bool tensors.
#
#   Complex tensors are "equal" if both the
#   real and complex parts (separately) are close. This is divergent from
#   torch.allclose's behavior, which compares the absolute values of the
#   complex numbers instead.
#
#   Using torch.allclose would be a less strict
#   comparison that would allow large complex values with
#   significant real or imaginary differences to be considered "equal,"
#   and would make setting rtol and atol for complex tensors distinct from
#   other tensor types.
#
#   Bool tensors are equal only if they are identical, regardless of
#   the rtol and atol values.
#
#   The `equal_nan` can be True or False, which maps to the True or False
#   in `torch.allclose`. `equal_nan` can also be "relaxed", which means
#   the complex will be compared in the relaxed mode:
#       2 + nan j == 3 + nan j ---> False when equal_nan=True
#                                   True when equal_nan="relaxed"
def _compare_tensors_internal(a: torch.Tensor, b: torch.Tensor, *, rtol, atol, equal_nan: Union[str, bool]) -> _compare_return_type:
    assert equal_nan in {True, False, "relaxed"}
    debug_msg : Optional[str]
    # Integer (including bool) comparisons are identity comparisons
    # when rtol is zero and atol is less than one
    if (
        (is_integral(a.dtype) and rtol == 0 and atol < 1)
        or a.dtype is torch.bool
        or is_quantized(a.dtype)
    ):
        if (a == b).all().item():
            return (True, None)

        # Gathers debug info for failed integer comparison
        # NOTE: converts to long to correctly represent differences
        # (especially between uint8 tensors)
        identity_mask = a != b
        a_flat = a.to(torch.long).flatten()
        b_flat = b.to(torch.long).flatten()
        count_non_identical = torch.sum(identity_mask, dtype=torch.long)
        diff = torch.abs(a_flat - b_flat)
        greatest_diff_index = torch.argmax(diff)
        debug_msg = ("Found {0} different element(s) (out of {1}), with the greatest "
                     "difference of {2} ({3} vs. {4}) occuring at index "
                     "{5}.".format(count_non_identical.item(),
                                   a.numel(),
                                   diff[greatest_diff_index],
                                   a_flat[greatest_diff_index],
                                   b_flat[greatest_diff_index],
                                   _unravel_index(greatest_diff_index, a.shape)))
        return (False, debug_msg)

    # Compares complex tensors' real and imaginary parts separately.
    # (see NOTE Test Framework Tensor "Equality")
    if a.is_complex():
        if equal_nan == "relaxed":
            a = a.clone()
            b = b.clone()
            a.real[a.imag.isnan()] = math.nan
            a.imag[a.real.isnan()] = math.nan
            b.real[b.imag.isnan()] = math.nan
            b.imag[b.real.isnan()] = math.nan

        real_result, debug_msg = _compare_tensors_internal(a.real, b.real,
                                                           rtol=rtol, atol=atol,
                                                           equal_nan=equal_nan)

        if not real_result:
            debug_msg = "Real parts failed to compare as equal! " + cast(str, debug_msg)
            return (real_result, debug_msg)

        imag_result, debug_msg = _compare_tensors_internal(a.imag, b.imag,
                                                           rtol=rtol, atol=atol,
                                                           equal_nan=equal_nan)

        if not imag_result:
            debug_msg = "Imaginary parts failed to compare as equal! " + cast(str, debug_msg)
            return (imag_result, debug_msg)

        return (True, None)

    # All other comparisons use torch.allclose directly
    if torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=(equal_nan in {"relaxed", True})):
        return (True, None)

    # Gathers debug info for failed float tensor comparison
    # NOTE: converts to float64 to best represent differences
    a_flat = a.to(torch.float64).flatten()
    b_flat = b.to(torch.float64).flatten()
    diff = torch.abs(a_flat - b_flat)

    # Masks close values
    # NOTE: this avoids (inf - inf) oddities when computing the difference
    close = torch.isclose(a_flat, b_flat, rtol, atol, (equal_nan in {"relaxed", True}))
    diff[close] = 0
    nans = torch.isnan(diff)
    num_nans = nans.sum()

    outside_range = (diff > (atol + rtol * torch.abs(b_flat))) | (diff == math.inf)
    count_outside_range = torch.sum(outside_range, dtype=torch.long)
    greatest_diff_index = torch.argmax(diff)
    debug_msg = ("With rtol={0} and atol={1}, found {2} element(s) (out of {3}) whose "
                 "difference(s) exceeded the margin of error (including {4} nan comparisons). "
                 "The greatest difference was {5} ({6} vs. {7}), which "
                 "occurred at index {8}.".format(rtol, atol,
                                                 count_outside_range + num_nans,
                                                 a.numel(),
                                                 num_nans,
                                                 diff[greatest_diff_index],
                                                 a_flat[greatest_diff_index],
                                                 b_flat[greatest_diff_index],
                                                 _unravel_index(greatest_diff_index, a.shape)))
    return (False, debug_msg)

# Checks if two scalars are equal(-ish), returning (True, None)
# when they are and (False, debug_msg) when they are not.
def _compare_scalars_internal(a, b, *, rtol: float, atol: float, equal_nan: Union[str, bool]) -> _compare_return_type:
    def _helper(a, b, s) -> _compare_return_type:
        # Short-circuits on identity
        if a == b or ((equal_nan in {"relaxed", True}) and a != a and b != b):
            return (True, None)

        # Special-case for NaN comparisions when equal_nan=False
        if not (equal_nan in {"relaxed", True}) and (a != a or b != b):
            msg = ("Found {0} and {1} while comparing" + s + "and either one "
                   "is nan and the other isn't, or both are nan and "
                   "equal_nan is False").format(a, b)
            return (False, msg)

        diff = abs(a - b)
        allowed_diff = atol + rtol * abs(b)
        result = diff <= allowed_diff

        # Special-case for infinity comparisons
        # NOTE: if b is inf then allowed_diff will be inf when rtol is not 0
        if ((math.isinf(a) or math.isinf(b)) and a != b):
            result = False

        msg = None
        if not result:
            msg = ("Comparing" + s + "{0} and {1} gives a "
                   "difference of {2}, but the allowed difference "
                   "with rtol={3} and atol={4} is "
                   "only {5}!").format(a, b, diff,
                                       rtol, atol, allowed_diff)

        return result, msg

    if isinstance(a, complex) or isinstance(b, complex):
        a = complex(a)
        b = complex(b)

        if equal_nan == "relaxed":
            if cmath.isnan(a) and cmath.isnan(b):
                return (True, None)

        result, msg = _helper(a.real, b.real, " the real part ")

        if not result:
            return (False, msg)

        return _helper(a.imag, b.imag, " the imaginary part ")

    return _helper(a, b, " ")

def assert_allclose(actual, expected, rtol=None, atol=None, equal_nan=True, msg='') -> None:
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual)
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)
    if expected.shape != actual.shape:
        raise AssertionError("expected tensor shape {0} doesn't match with actual tensor "
                             "shape {1}!".format(expected.shape, actual.shape))
    if rtol is None or atol is None:
        if rtol is not None or atol is not None:
            raise ValueError("rtol and atol must both be specified or both be unspecified")
        rtol, atol = _get_default_tolerance(actual, expected)

    result, debug_msg = _compare_tensors_internal(actual, expected,
                                                  rtol=rtol, atol=atol,
                                                  equal_nan=equal_nan)

    if result:
        return

    if msg is None or msg == '':
        msg = debug_msg

    raise AssertionError(msg)

def make_non_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() <= 1:  # can't make non-contiguous
        return tensor.clone()
    osize = list(tensor.size())

    # randomly inflate a few dimensions in osize
    for _ in range(2):
        dim = random.randint(0, len(osize) - 1)
        add = random.randint(4, 15)
        osize[dim] = osize[dim] + add

    # narrow doesn't make a non-contiguous tensor if we only narrow the 0-th dimension,
    # (which will always happen with a 1-dimensional tensor), so let's make a new
    # right-most dimension and cut it off

    input = tensor.new(torch.Size(osize + [random.randint(2, 3)]))
    input = input.select(len(input.size()) - 1, random.randint(0, 1))
    # now extract the input of correct size from 'input'
    for i in range(len(osize)):
        if input.size(i) != tensor.size(i):
            bounds = random.randint(1, input.size(i) - tensor.size(i))
            input = input.narrow(i, bounds, tensor.size(i))

    input.copy_(tensor)

    # Use .data here to hide the view relation between input and other temporary Tensors
    return input.data


# Functions and classes for describing the dtypes a function supports
# NOTE: these helpers should correspond to PyTorch's C++ dispatch macros

# Verifies each given dtype is a torch.dtype
def _validate_dtypes(*dtypes):
    for dtype in dtypes:
        assert isinstance(dtype, torch.dtype)
    return dtypes

# class for tuples corresponding to a PyTorch dispatch macro
class _dispatch_dtypes(tuple):
    def __add__(self, other):
        assert isinstance(other, tuple)
        return _dispatch_dtypes(tuple.__add__(self, other))

_floating_types = _dispatch_dtypes((torch.float32, torch.float64))
def floating_types():
    return _floating_types

_floating_types_and_half = _floating_types + (torch.half,)
def floating_types_and_half():
    return _floating_types_and_half

def floating_types_and(*dtypes):
    return _floating_types + _validate_dtypes(*dtypes)

_floating_and_complex_types = _floating_types + (torch.cfloat, torch.cdouble)
def floating_and_complex_types():
    return _floating_and_complex_types

def floating_and_complex_types_and(*dtypes):
    return _floating_and_complex_types + _validate_dtypes(*dtypes)

_integral_types = _dispatch_dtypes((torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64))
def integral_types():
    return _integral_types

def integral_types_and(*dtypes):
    return _integral_types + _validate_dtypes(*dtypes)

_all_types = _floating_types + _integral_types
def all_types():
    return _all_types

def all_types_and(*dtypes):
    return _all_types + _validate_dtypes(*dtypes)

_complex_types = (torch.cfloat, torch.cdouble)
def complex_types():
    return _complex_types

_all_types_and_complex = _all_types + _complex_types
def all_types_and_complex():
    return _all_types_and_complex

def all_types_and_complex_and(*dtypes):
    return _all_types_and_complex + _validate_dtypes(*dtypes)

_all_types_and_half = _all_types + (torch.half,)
def all_types_and_half():
    return _all_types_and_half

def get_all_dtypes(include_half=True,
                   include_bfloat16=True,
                   include_bool=True,
                   include_complex=True,
                   include_complex32=False
                   ) -> List[torch.dtype]:
    dtypes = get_all_int_dtypes() + get_all_fp_dtypes(include_half=include_half, include_bfloat16=include_bfloat16)
    if include_bool:
        dtypes.append(torch.bool)
    if include_complex:
        dtypes += get_all_complex_dtypes(include_complex32)
    return dtypes

def get_all_math_dtypes(device) -> List[torch.dtype]:
    return get_all_int_dtypes() + get_all_fp_dtypes(include_half=device.startswith('cuda'),
                                                    include_bfloat16=False) + get_all_complex_dtypes()

def get_all_complex_dtypes(include_complex32=False) -> List[torch.dtype]:
    return [torch.complex32, torch.complex64, torch.complex128] if include_complex32 else [torch.complex64, torch.complex128]


def get_all_int_dtypes() -> List[torch.dtype]:
    return [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def get_all_fp_dtypes(include_half=True, include_bfloat16=True) -> List[torch.dtype]:
    dtypes = [torch.float32, torch.float64]
    if include_half:
        dtypes.append(torch.float16)
    if include_bfloat16:
        dtypes.append(torch.bfloat16)
    return dtypes


def get_all_device_types() -> List[str]:
    return ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']

# 'dtype': (rtol, atol)
_default_tolerances = {
    'float64': (1e-5, 1e-8),  # NumPy default
    'float32': (1e-4, 1e-5),  # This may need to be changed
    'float16': (1e-3, 1e-3),  # This may need to be changed
}


def _get_default_tolerance(a, b=None) -> Tuple[float, float]:
    if b is None:
        dtype = str(a.dtype).split('.')[-1]  # e.g. "float32"
        return _default_tolerances.get(dtype, (0, 0))
    a_tol = _get_default_tolerance(a)
    b_tol = _get_default_tolerance(b)
    return (max(a_tol[0], b_tol[0]), max(a_tol[1], b_tol[1]))
