import collections
import warnings
from functools import partial, wraps
from typing import Sequence

import numpy as np

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_dtype import (
    _dispatch_dtypes,
    all_types,
    all_types_and,
    all_types_and_complex,
    all_types_and_complex_and,
    all_types_and_half,
    complex_types,
    floating_and_complex_types,
    floating_and_complex_types_and,
    floating_types,
    floating_types_and,
    floating_types_and_half,
    integral_types,
    integral_types_and,
)
from torch.testing._internal.common_utils import torch_to_numpy_dtype_dict


COMPLETE_DTYPES_DISPATCH = (
    all_types,
    all_types_and_complex,
    all_types_and_half,
    floating_types,
    floating_and_complex_types,
    floating_types_and_half,
    integral_types,
    complex_types,
)

EXTENSIBLE_DTYPE_DISPATCH = (
    all_types_and_complex_and,
    floating_types_and,
    floating_and_complex_types_and,
    integral_types_and,
    all_types_and,
)

# Better way to acquire devices?
DEVICES = ["cpu"] + (["cuda"] if TEST_CUDA else [])


class _dynamic_dispatch_dtypes(_dispatch_dtypes):
    # Class to tag the dynamically generated types.
    pass


def get_supported_dtypes(op, sample_inputs_fn, device_type):
    # Returns the supported dtypes for the given operator and device_type pair.
    assert device_type in ["cpu", "cuda"]
    if not TEST_CUDA and device_type == "cuda":
        warnings.warn(
            "WARNING: CUDA is not available, empty_dtypes dispatch will be returned!",
            stacklevel=TO_BE_DETERMINED,
        )
        return _dynamic_dispatch_dtypes(())

    supported_dtypes = set()
    for dtype in all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half):
        try:
            samples = sample_inputs_fn(op, device_type, dtype, False)
        except RuntimeError:
            # If `sample_inputs_fn` doesn't support sampling for a given
            # `dtype`, we assume that the `dtype` is not supported.
            # We raise a warning, so that user knows that this was the case
            # and can investigate if there was an issue with the `sample_inputs_fn`.
            warnings.warn(
                f"WARNING: Unable to generate sample for device:{device_type} and dtype:{dtype}",
                stacklevel=TO_BE_DETERMINED,
            )
            continue

        # We assume the dtype is supported
        # only if all samples pass for the given dtype.
        supported = True
        for sample in samples:
            try:
                op(sample.input, *sample.args, **sample.kwargs)
            except RuntimeError as re:
                # dtype is not supported
                supported = False
                break

        if supported:
            supported_dtypes.add(dtype)

    return _dynamic_dispatch_dtypes(supported_dtypes)


def dtypes_dispatch_hint(dtypes):
    # Function returns the appropriate dispatch function (from COMPLETE_DTYPES_DISPATCH and EXTENSIBLE_DTYPE_DISPATCH)
    # and its string representation for the passed `dtypes`.
    return_type = collections.namedtuple("return_type", "dispatch_fn dispatch_fn_str")

    # CUDA is not available, dtypes will be empty.
    if len(dtypes) == 0:
        return return_type((), str(tuple()))

    set_dtypes = set(dtypes)
    for dispatch in COMPLETE_DTYPES_DISPATCH:
        # Short circuit if we get an exact match.
        if set(dispatch()) == set_dtypes:
            return return_type(dispatch, dispatch.__name__ + "()")

    chosen_dispatch = None
    chosen_dispatch_score = 0.0
    for dispatch in EXTENSIBLE_DTYPE_DISPATCH:
        dispatch_dtypes = set(dispatch())
        if not dispatch_dtypes.issubset(set_dtypes):
            continue

        score = len(dispatch_dtypes)
        if score > chosen_dispatch_score:
            chosen_dispatch_score = score
            chosen_dispatch = dispatch

    # If user passed dtypes which are lower than the lowest
    # dispatch type available (not likely but possible in code path).
    if chosen_dispatch is None:
        return return_type((), str(dtypes))

    return return_type(
        partial(dispatch, *tuple(set(dtypes) - set(dispatch()))),
        dispatch.__name__ + str(tuple(set(dtypes) - set(dispatch()))),
    )


def is_dynamic_dtype_set(op):
    # Detect if the OpInfo entry acquired dtypes dynamically
    # using `get_supported_dtypes`.
    return op.dynamic_dtypes


def str_format_dynamic_dtype(op):
    fmt_str = f"""
        OpInfo({op.name},
               dtypes={dtypes_dispatch_hint(op.dtypes).dispatch_fn_str},
               dtypesIfCUDA={dtypes_dispatch_hint(op.dtypesIfCUDA).dispatch_fn_str},
        )
        """

    return fmt_str


def np_unary_ufunc_integer_promotion_wrapper(fn):
    # Wrapper that passes PyTorch's default scalar
    #   type as an argument to the wrapped NumPy
    #   unary ufunc when given an integer input.
    #   This mimicks PyTorch's integer->floating point
    #   type promotion.
    #
    # This is necessary when NumPy promotes
    #   integer types to double, since PyTorch promotes
    #   integer types to the default scalar type.

    # Helper to determine if promotion is needed
    def is_integral(dtype):
        return dtype in [
            np.bool_,
            bool,
            np.uint8,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ]

    @wraps(fn)
    def wrapped_fn(x):
        # As the default dtype can change, acquire it when function is called.
        # NOTE: Promotion in PyTorch is from integer types to the default dtype
        np_dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]

        if is_integral(x.dtype):
            return fn(x.astype(np_dtype))
        return fn(x)

    return wrapped_fn


def reference_reduction_numpy(f, supports_keepdims=True):
    """Wraps a NumPy reduction operator.

    The wrapper function will forward dim, keepdim, mask, and identity
    kwargs to the wrapped function as the NumPy equivalent axis,
    keepdims, where, and initiak kwargs, respectively.

    Args:
        f: NumPy reduction operator to wrap
        supports_keepdims (bool, optional): Whether the NumPy operator accepts
            keepdims parameter. If it does not, the wrapper will manually unsqueeze
            the reduced dimensions if it was called with keepdim=True. Defaults to True.

    Returns:
        Wrapped function

    """

    @wraps(f)
    def wrapper(x: np.ndarray, *args, **kwargs):
        # Copy keys into a set
        keys = set(kwargs.keys())

        dim = kwargs.pop("dim", None)
        keepdim = kwargs.pop("keepdim", False)

        if "dim" in keys:
            dim = tuple(dim) if isinstance(dim, Sequence) else dim

            # NumPy reductions don't accept dim=0 for scalar inputs
            # so we convert it to None if and only if dim is equivalent
            if x.ndim == 0 and dim in {0, -1, (0,), (-1,)}:
                kwargs["axis"] = None
            else:
                kwargs["axis"] = dim

        if "keepdim" in keys and supports_keepdims:
            kwargs["keepdims"] = keepdim

        if "mask" in keys:
            mask = kwargs.pop("mask")
            if mask is not None:
                assert mask.layout == torch.strided
                kwargs["where"] = mask.cpu().numpy()

        if "identity" in keys:
            identity = kwargs.pop("identity")
            if identity is not None:
                if identity.dtype is torch.bfloat16:
                    identity = identity.cpu().to(torch.float32)
                else:
                    identity = identity.cpu()
                kwargs["initial"] = identity.numpy()

        result = f(x, *args, **kwargs)

        # Unsqueeze reduced dimensions if NumPy does not support keepdims
        if keepdim and not supports_keepdims and x.ndim > 0:
            dim = list(range(x.ndim)) if dim is None else dim
            result = np.expand_dims(result, dim)

        return result

    return wrapper


def prod_numpy(a, *args, **kwargs):
    """
    The function will call np.prod with type as np.int64 if the input type
    is int or uint64 if is uint. This is necessary because windows np.prod uses by default
    int32 while on linux it uses int64.
    This is for fixing integer overflow https://github.com/pytorch/pytorch/issues/77320

    Returns:
        np.prod of input
    """
    if "dtype" not in kwargs:
        if np.issubdtype(a.dtype, np.signedinteger):
            a = a.astype(np.int64)
        elif np.issubdtype(a.dtype, np.unsignedinteger):
            a = a.astype(np.uint64)

    fn = reference_reduction_numpy(np.prod)
    return fn(a, *args, **kwargs)
