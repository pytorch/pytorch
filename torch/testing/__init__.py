"""
The testing package contains testing-specific utilities.
"""

import torch
import random
import math
from functools import partial

FileCheck = torch._C.FileCheck

__all__ = [
    'assert_allclose', 'make_non_contiguous', 'rand_like', 'randn_like'
]

rand_like = torch.rand_like
randn_like = torch.randn_like

# Datastructures and helpers for returning a collection of appropriate
# tensors to use during testing.
# NOTE: Intended to eventually be a replacement for
#   most ad hoc and historic test tensor construction methods like _make_tensor
#   and _make_tensors in test_torch.py.
_complex_nonfinites = [complex(float('inf'), float('inf')),
                       complex(float('inf'), float('-inf')),
                       complex(float('inf'), float('nan')),
                       complex(float('inf'), 0),
                       complex(float('-inf'), float('inf')),
                       complex(float('-inf'), float('-inf')),
                       complex(float('-inf'), float('nan')),
                       complex(float('-inf'), 0),
                       complex(float('nan'), float('inf')),
                       complex(float('nan'), float('-inf')),
                       complex(float('nan'), float('nan')),
                       complex(float('nan'), 0),
                       complex(0, float('inf')),
                       complex(0, float('-inf')),
                       complex(0, float('nan'))]
_complex_defaults = [complex(0, 0),
                     complex(1, 0), complex(-1, 0), complex(0, 1), complex(0, -1),
                     complex(math.pi, 0), complex(-math.pi), complex(0, math.pi), complex(0, -math.pi),
                     complex(math.pi, math.pi), complex(math.pi, -math.pi),
                     complex(-math.pi, math.pi), complex(-math.pi, -math.pi)]

_int_defaults = [0, 1, 127]
_bool_defaults = [True, False]

_float_nonfinites = [float('inf'), float('-inf'), float('nan')]
_float_defaults = [0., -1., 1., -math.pi, math.pi]

_nelems_large = 65
_nelems_small = 4

# Helper for fixtures. See fixtures documentation.
def _fixtures_helper(vals, device, dtype, *,
                     type_filter, tensor_factory):
    # Filters mismatched values
    vals = list(set([v for v in vals if isinstance(v, type_filter)]))

    v = torch.tensor(vals, device=device, dtype=dtype)
    results = []

    # Creates "large" tensor
    # NOTE: the large tensor is filled with the values in vals followed by
    #  random values.
    large_t = None
    if len(vals) > _nelems_large:
        large_t = v
    else:
        large_t = tensor_factory((_nelems_large,), device=device, dtype=dtype)
        large_t[0:len(vals)] = v
        results.append(large_t)

    # Creates scalar tensors
    scalars = []
    if len(vals) == 0:
        scalars.append(tensor_factory((1,), device=device, dtype=dtype))
    else:
        for v in vals:
            scalars.append(torch.tensor((v), device=device, dtype=dtype))

    return results

# TODO: support division generation by bounding away from zero (min abs)
# TODO: support nDims parameter (produces square tensors)
# TODO: support shape parameter
# TODO: support range parameter (e.g. values between 100 and 200)
# TODO: support noncontiguous outputs
# TODO: allow disabling scalar tensor fixtures
# TODO: replace test_torch.py's _make_tensor and _make_tensors
# Returns a list of tensors suitable for testing. The list includes a larger
#   tensor suitable for validating vectorization pathways as well as at least
#   one zero-dim "scalar" tensor to test non-vectorized codepaths.
#
#   if include_nonfinite is True then nonfinite values like infinity are in the tensors
#   if include_defaults is True then commonly used values like 0 and 1 are in the tensors
def fixtures(vals, device, dtype, *, include_nonfinite=True, include_defaults=True):
    assert dtype is not torch.bfloat16, "bfloat16 fixtures not yet supported"
    assert dtype is not torch.float16, "float16 fixtures not yet supported"

    vals = list(vals)

    non_finites = None
    defaults = None
    if dtype.is_floating_point:
        type_filter = float
        tensor_factory = torch.randn
        non_finites = _float_nonfinites
        defaults = _float_defaults
    elif dtype.is_complex:
        type_filter = complex
        tensor_factory = torch.randn
        non_finites = _complex_nonfinites
        include_defaults = _complex_defaults
    else:
        type_filter = int
        tensor_factory = partial(torch.randint, -9, 10)
        defaults = _int_defaults

        # Specializations for unsigned types
        if dtype is torch.uint8:
            tensor_factory = partial(torch.randint, 0, 10)
        elif dtype is torch.bool:
            tensor_factory = partial(torch.randint, 0, 2)
            defaults = _bool_defaults

    # Includes nonfinites (if requested)
    if include_nonfinite and non_finites is not None:
        vals.extend(non_finites)
    # Includes defaults (if requested)
    if include_defaults and defaults is not None:
        vals.extend(defaults)

    return _fixtures_helper(vals, device, dtype,
                            type_filter=type_filter,
                            tensor_factory=tensor_factory)

# Helper function that returns True when the dtype is an integral dtype,
# False otherwise.
# TODO: implement numpy-like issubdtype
def is_integral(dtype):
    # Skip complex/quantized types
    dtypes = [x for x in get_all_dtypes() if x not in get_all_complex_dtypes()]
    return dtype in dtypes and not dtype.is_floating_point

# Helper function that maps a flattened index back into the given shape
# TODO: consider adding torch.unravel_index
def _unravel_index(flat_index, shape):
    res = []

    # Short-circuits on zero dim tensors
    if shape == torch.Size([]):
        return 0

    for size in shape[::-1]:
        res.append(int(flat_index % size))
        flat_index = int(flat_index // size)

    if len(res) == 1:
        return res[0]

    return tuple(res[::-1])

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
def _compare_tensors_internal(a, b, *, rtol, atol, equal_nan):
    # Integer (including bool) comparisons are identity comparisons
    # when rtol is zero and atol is less than one
    if (is_integral(a.dtype) and rtol == 0 and atol < 1) or a.dtype is torch.bool:
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
        float_dtype = torch.float32 if a.dtype == torch.complex64 else torch.float64
        a_real = a.copy_real().to(float_dtype)
        b_real = b.copy_real().to(float_dtype)
        real_result, debug_msg = _compare_tensors_internal(a_real, b_real,
                                                           rtol=rtol, atol=atol,
                                                           equal_nan=equal_nan)

        if not real_result:
            debug_msg = "Real parts failed to compare as equal! " + debug_msg
            return (real_result, debug_msg)

        a_imag = a.copy_imag().to(float_dtype)
        b_imag = b.copy_imag().to(float_dtype)
        imag_result, debug_msg = _compare_tensors_internal(a_imag, b_imag,
                                                           rtol=rtol, atol=atol,
                                                           equal_nan=equal_nan)

        if not imag_result:
            debug_msg = "Imaginary parts failed to compare as equal! " + debug_msg
            return (imag_result, debug_msg)

        return (True, None)

    # All other comparisons use torch.allclose directly
    if torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan):
        return (True, None)

    # Gathers debug info for failed float tensor comparison
    # NOTE: converts to float64 to best represent differences
    a_flat = a.to(torch.float64).flatten()
    b_flat = b.to(torch.float64).flatten()
    diff = torch.abs(a_flat - b_flat)

    # Masks close values
    # NOTE: this avoids (inf - inf) oddities when computing the difference
    close = torch.isclose(a_flat, b_flat, rtol, atol, equal_nan)
    diff[close] = 0
    nans = torch.isnan(diff)
    num_nans = nans.sum()

    outside_range = diff > (atol + rtol * torch.abs(b_flat))
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
def _compare_scalars_internal(a, b, *, rtol, atol, equal_nan):
    def _helper(a, b, s):
        # Short-circuits on identity
        if a == b or (equal_nan and a != a and b != b):
            return (True, None)

        # Special-case for NaN comparisions when equal_nan=False
        if not equal_nan and (a != a or b != b):
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

        result, msg = _helper(a.real, b.real, " the real part ")

        if not result:
            return (False, msg)

        return _helper(a.imag, b.imag, " the imaginary part ")

    return _helper(a, b, " ")

def assert_allclose(actual, expected, rtol=None, atol=None, equal_nan=True, msg=''):
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual)
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)
    if expected.shape != actual.shape:
        expected = expected.expand_as(actual)
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

def make_non_contiguous(tensor):
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


def get_all_dtypes(include_half=True, include_bfloat16=True, include_bool=True, include_complex=True):
    dtypes = get_all_int_dtypes() + get_all_fp_dtypes(include_half=include_half, include_bfloat16=include_bfloat16)
    if include_bool:
        dtypes.append(torch.bool)
    if include_complex:
        dtypes += get_all_complex_dtypes()
    return dtypes


def get_all_math_dtypes(device):
    return get_all_int_dtypes() + get_all_fp_dtypes(include_half=device.startswith('cuda'),
                                                    include_bfloat16=False) + get_all_complex_dtypes()


def get_all_complex_dtypes():
    return [torch.complex64, torch.complex128]


def get_all_int_dtypes():
    return [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def get_all_fp_dtypes(include_half=True, include_bfloat16=True):
    dtypes = [torch.float32, torch.float64]
    if include_half:
        dtypes.append(torch.float16)
    if include_bfloat16:
        dtypes.append(torch.bfloat16)
    return dtypes


def get_all_device_types():
    return ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']

# 'dtype': (rtol, atol)
_default_tolerances = {
    'float64': (1e-5, 1e-8),  # NumPy default
    'float32': (1e-4, 1e-5),  # This may need to be changed
    'float16': (1e-3, 1e-3),  # This may need to be changed
}


def _get_default_tolerance(a, b=None):
    if b is None:
        dtype = str(a.dtype).split('.')[-1]  # e.g. "float32"
        return _default_tolerances.get(dtype, (0, 0))
    a_tol = _get_default_tolerance(a)
    b_tol = _get_default_tolerance(b)
    return (max(a_tol[0], b_tol[0]), max(a_tol[1], b_tol[1]))
