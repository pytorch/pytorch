"""This module exist to be able to deprecate functions publicly without doing so internally. The deprecated
public versions are defined in torch.testing._deprecated and exposed from torch.testing. The non-deprecated internal
versions should be imported from torch.testing._internal
"""

from typing import List

import torch

__all_dtype_getters__ = [
    "_validate_dtypes",
    "_dispatch_dtypes",
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
    "get_all_fp_dtypes",
    "get_all_int_dtypes",
    "get_all_math_dtypes",
    "integral_types",
    "integral_types_and",
]

__all__ = [
    *__all_dtype_getters__,
    "get_all_device_types",
]

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

# The functions below are used for convenience in our test suite and thus have no corresponding C++ dispatch macro

def get_all_dtypes(include_half=True,
                   include_bfloat16=True,
                   include_bool=True,
                   include_complex=True
                   ) -> List[torch.dtype]:
    dtypes = get_all_int_dtypes() + get_all_fp_dtypes(include_half=include_half, include_bfloat16=include_bfloat16)
    if include_bool:
        dtypes.append(torch.bool)
    if include_complex:
        dtypes += get_all_complex_dtypes()
    return dtypes

def get_all_math_dtypes(device) -> List[torch.dtype]:
    return get_all_int_dtypes() + get_all_fp_dtypes(include_half=device.startswith('cuda'),
                                                    include_bfloat16=False) + get_all_complex_dtypes()

def get_all_complex_dtypes() -> List[torch.dtype]:
    return [torch.complex64, torch.complex128]


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
