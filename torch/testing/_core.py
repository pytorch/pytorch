"""
The testing package contains testing-specific utilities.
"""

import torch
import random
import math
import cmath
from typing import cast, List, Optional, Tuple, Union
import operator

FileCheck = torch._C.FileCheck

__all__ = [
    "FileCheck",
    "all_types",
    "all_types_and",
    "all_types_and_complex",
    "all_types_and_complex_and",
    "all_types_and_half",
    "complex_types",
    "empty_types",
    "floating_and_complex_types",
    "floating_and_complex_types_and",
    "floating_types",
    "floating_types_and",
    "double_types",
    "floating_types_and_half",
    "get_all_complex_dtypes",
    "get_all_dtypes",
    "get_all_device_types",
    "get_all_fp_dtypes",
    "get_all_int_dtypes",
    "get_all_math_dtypes",
    "integral_types",
    "integral_types_and",
    "make_non_contiguous",
    "make_tensor"
]

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
            if rtol == 0 and atol == 0:
                msg = f"{a} != {b}"
            else:
                msg = (
                    f"Comparing{s}{a} and {b} gives a "
                    f"difference of {diff}, but the allowed difference "
                    f"with rtol={rtol} and atol={atol} is "
                    f"only {allowed_diff}!"
                )
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


def make_tensor(size, device: torch.device, dtype: torch.dtype, *, low=None, high=None,
                requires_grad: bool = False, noncontiguous: bool = False,
                exclude_zero: bool = False) -> torch.Tensor:
    """ Creates a random tensor with the given :attr:`size`, :attr:`device` and :attr:`dtype`.

        The function comes with other optional arguments to allow random tensor generation for the user's needs.

        If :attr:`low` and :attr:`high` are not passed, following default values are considered depending on the given
        :attr:`dtype`:

            * boolean type: `low` = 0, `high` = 2
            * uint8 type: `low` = 0, `high` = 9
            * floating and integral types: `low` = -9 and `high` = 9
            * complex types, for each real and imaginary part: `low` = -9, `high` = 9

        If :attr:`low` and :attr:`high` are passed, they are considered only if they are within the
        limit of the :attr:`dtype`. Following are a few conditions that are taken care of:

            * If :attr:`low` and/or :attr:`high` are specified and within dtype limits: the values are taken as they were.
            * If :attr:`low` and/or :attr:`high` are specified but exceed the limits:
                :attr:`dtype` limits are considered instead
            * If :attr:`low` is ``-inf`` and/or :attr:`high` is ``inf``:
                :attr:`dtype` limits are considered instead
            * If :attr:`low` is ``inf`` or ``nan`` and/or :attr:`high` is ``-inf`` or nan:
                A `ValueError` is raised, since these are invalid values for the range of output tensor.

        If :attr:`noncontiguous` is ``True``, a noncontiguous tensor with the given size will be returned unless the
        size specifies a tensor with a 1 or 0 elements in which case the noncontiguous parameter is ignored because
        it is not possible to create a noncontiguous Tensor with a single element.

        If :attr:`exclude_zero` is ``True`` (default is ``False``), all the values matching to zero in
        the created tensor are replaced with a ``tiny`` (smallest positive representable number) value if floating type,
        [``tiny`` + ``tiny``.j] if complex type and ``1`` if integer/boolean type.
    """
    def _modify_low_high(low, high, lowest, highest, default_low, default_high, dtype):
        """
        Modifies (and raises ValueError when appropriate) low and high values given by the user (input_low, input_high) if required.
        """
        def clamp(a, l, h):
            return min(max(a, l), h)

        low = low if low is not None else default_low
        high = high if high is not None else default_high

        # Checks for error cases
        if low != low or high != high:
            raise ValueError("make_tensor: one of low or high was NaN!")
        if low > high:
            raise ValueError("make_tensor: low must be weakly less than high!")

        low = clamp(low, lowest, highest)
        high = clamp(high, lowest, highest)

        if dtype in integral_types():
            return math.floor(low), math.ceil(high)

        return low, high

    if dtype is torch.bool:
        result = torch.randint(0, 2, size, device=device, dtype=dtype)
    elif dtype is torch.uint8:
        ranges = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges[0], ranges[1], 0, 9, dtype)
        result = torch.randint(low, high, size, device=device, dtype=dtype)
    elif dtype in integral_types():
        ranges = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges[0], ranges[1], -9, 9, dtype)
        result = torch.randint(low, high, size, device=device, dtype=dtype)
    elif dtype in floating_types_and(torch.half, torch.bfloat16):
        ranges_floats = (torch.finfo(dtype).min, torch.finfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges_floats[0], ranges_floats[1], -9, 9, dtype)
        rand_val = torch.rand(size, device=device, dtype=dtype)
        result = high * rand_val + low * (1 - rand_val)
    else:
        assert dtype in complex_types()
        float_dtype = torch.float if dtype is torch.cfloat else torch.double
        ranges_floats = (torch.finfo(float_dtype).min, torch.finfo(float_dtype).max)
        low, high = _modify_low_high(low, high, ranges_floats[0], ranges_floats[1], -9, 9, dtype)
        real_rand_val = torch.rand(size, device=device, dtype=float_dtype)
        imag_rand_val = torch.rand(size, device=device, dtype=float_dtype)
        real = high * real_rand_val + low * (1 - real_rand_val)
        imag = high * imag_rand_val + low * (1 - imag_rand_val)
        result = torch.complex(real, imag)

    if noncontiguous and result.numel() > 1:
        result = torch.repeat_interleave(result, 2, dim=-1)
        result = result[..., ::2]

    if exclude_zero:
        if dtype in integral_types() or dtype is torch.bool:
            replace_with = torch.tensor(1, device=device, dtype=dtype)
        elif dtype in floating_types_and(torch.half, torch.bfloat16):
            replace_with = torch.tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)
        elif dtype in complex_types():
            float_dtype = torch.float if dtype is torch.cfloat else torch.double
            float_eps = torch.tensor(torch.finfo(float_dtype).tiny, device=device, dtype=float_dtype)
            replace_with = torch.complex(float_eps, float_eps)
        else:
            raise ValueError(f"Invalid dtype passed, supported dtypes are: {get_all_dtypes()}")
        result[result == 0] = replace_with

    if dtype in floating_types_and(torch.half, torch.bfloat16) or\
       dtype in complex_types():
        result.requires_grad = requires_grad

    return result


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

_empty_types = _dispatch_dtypes(())
def empty_types():
    return _empty_types

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

_double_types = _dispatch_dtypes((torch.float64, torch.complex128))
def double_types():
    return _double_types

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

_complex_types = _dispatch_dtypes((torch.cfloat, torch.cdouble))
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
