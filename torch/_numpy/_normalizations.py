""" "Normalize" arguments: convert array_likes to tensors, dtypes to torch dtypes and so on.
"""
from __future__ import annotations

import functools
import inspect
import operator
import typing

import torch

from . import _dtypes, _dtypes_impl, _util

ArrayLike = typing.TypeVar("ArrayLike")
Scalar = typing.Union[int, float, complex, bool]
ArrayLikeOrScalar = typing.Union[ArrayLike, Scalar]

DTypeLike = typing.TypeVar("DTypeLike")
AxisLike = typing.TypeVar("AxisLike")
NDArray = typing.TypeVar("NDarray")
CastingModes = typing.TypeVar("CastingModes")
KeepDims = typing.TypeVar("KeepDims")

# OutArray is to annotate the out= array argument.
#
# This one is special is several respects:
# First, It needs to be an NDArray, and we need to preserve the `result is out`
# semantics. Therefore, we cannot just extract the Tensor from the out array.
# So we never pass the out array to implementer functions and handle it in the
# `normalizer` below.
# Second, the out= argument can be either keyword or positional argument, and
# as a positional arg, it can be anywhere in the signature.
# To handle all this, we define a special `OutArray` annotation and dispatch on it.
#
OutArray = typing.TypeVar("OutArray")

try:
    from typing import NotImplementedType
except ImportError:
    NotImplementedType = typing.TypeVar("NotImplementedType")


def normalize_array_like(x, parm=None):
    from ._ndarray import asarray

    return asarray(x).tensor


def normalize_array_like_or_scalar(x, parm=None):
    if _dtypes_impl.is_scalar_or_symbolic(x):
        return x
    return normalize_array_like(x, parm)


def normalize_optional_array_like_or_scalar(x, parm=None):
    if x is None:
        return None
    return normalize_array_like_or_scalar(x, parm)


def normalize_optional_array_like(x, parm=None):
    # This explicit normalizer is needed because otherwise normalize_array_like
    # does not run for a parameter annotated as Optional[ArrayLike]
    return None if x is None else normalize_array_like(x, parm)


def normalize_seq_array_like(x, parm=None):
    return tuple(normalize_array_like(value) for value in x)


def normalize_dtype(dtype, parm=None):
    # cf _decorators.dtype_to_torch
    torch_dtype = None
    if dtype is not None:
        dtype = _dtypes.dtype(dtype)
        torch_dtype = dtype.torch_dtype
    return torch_dtype


def normalize_not_implemented(arg, parm):
    if arg != parm.default:
        raise NotImplementedError(f"'{parm.name}' parameter is not supported.")


def normalize_axis_like(arg, parm=None):
    from ._ndarray import ndarray

    if isinstance(arg, ndarray):
        arg = operator.index(arg)
    return arg


def normalize_ndarray(arg, parm=None):
    # check the arg is an ndarray, extract its tensor attribute
    if arg is None:
        return arg

    from ._ndarray import ndarray

    if not isinstance(arg, ndarray):
        raise TypeError(f"'{parm.name}' must be an array")
    return arg.tensor


def normalize_outarray(arg, parm=None):
    # almost normalize_ndarray, only return the array, not its tensor
    if arg is None:
        return arg

    from ._ndarray import ndarray

    if not isinstance(arg, ndarray):
        raise TypeError(f"'{parm.name}' must be an array")
    return arg


def normalize_casting(arg, parm=None):
    if arg not in ["no", "equiv", "safe", "same_kind", "unsafe"]:
        raise ValueError(
            f"casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe' (got '{arg}')"
        )
    return arg


normalizers = {
    "ArrayLike": normalize_array_like,
    "ArrayLikeOrScalar": normalize_array_like_or_scalar,
    "Optional[ArrayLike]": normalize_optional_array_like,
    "Sequence[ArrayLike]": normalize_seq_array_like,
    "Optional[ArrayLikeOrScalar]": normalize_optional_array_like_or_scalar,
    "Optional[NDArray]": normalize_ndarray,
    "Optional[OutArray]": normalize_outarray,
    "NDArray": normalize_ndarray,
    "Optional[DTypeLike]": normalize_dtype,
    "AxisLike": normalize_axis_like,
    "NotImplementedType": normalize_not_implemented,
    "Optional[CastingModes]": normalize_casting,
}


def maybe_normalize(arg, parm):
    """Normalize arg if a normalizer is registered."""
    normalizer = normalizers.get(parm.annotation, None)
    return normalizer(arg, parm) if normalizer else arg


# ### Return value helpers ###


def maybe_copy_to(out, result, promote_scalar_result=False):
    # NB: here out is either an ndarray or None
    if out is None:
        return result
    elif isinstance(result, torch.Tensor):
        if result.shape != out.shape:
            can_fit = result.numel() == 1 and out.ndim == 0
            if promote_scalar_result and can_fit:
                result = result.squeeze()
            else:
                raise ValueError(
                    f"Bad size of the out array: out.shape = {out.shape}"
                    f" while result.shape = {result.shape}."
                )
        out.tensor.copy_(result)
        return out
    elif isinstance(result, (tuple, list)):
        return type(result)(
            maybe_copy_to(o, r, promote_scalar_result) for o, r in zip(out, result)
        )
    else:
        raise AssertionError()  # We should never hit this path


def wrap_tensors(result):
    from ._ndarray import ndarray

    if isinstance(result, torch.Tensor):
        return ndarray(result)
    elif isinstance(result, (tuple, list)):
        result = type(result)(wrap_tensors(x) for x in result)
    return result


def array_or_scalar(values, py_type=float, return_scalar=False):
    if return_scalar:
        return py_type(values.item())
    else:
        from ._ndarray import ndarray

        return ndarray(values)


# ### The main decorator to normalize arguments / postprocess the output ###


def normalizer(_func=None, *, promote_scalar_result=False):
    def normalizer_inner(func):
        @functools.wraps(func)
        def wrapped(*args, **kwds):
            sig = inspect.signature(func)
            params = sig.parameters
            first_param = next(iter(params.values()))

            # NumPy's API does not have positional args before variadic positional args
            if first_param.kind == inspect.Parameter.VAR_POSITIONAL:
                args = [maybe_normalize(arg, first_param) for arg in args]
            else:
                # NB: extra unknown arguments: pass through, will raise in func(*args) below
                args = (
                    tuple(
                        maybe_normalize(arg, parm)
                        for arg, parm in zip(args, params.values())
                    )
                    + args[len(params.values()) :]
                )

            kwds = {
                name: maybe_normalize(arg, params[name]) if name in params else arg
                for name, arg in kwds.items()
            }

            result = func(*args, **kwds)

            # keepdims
            bound_args = None
            if "keepdims" in params and params["keepdims"].annotation == "KeepDims":
                # keepdims can be in any position so we need sig.bind
                bound_args = sig.bind(*args, **kwds).arguments
                if bound_args.get("keepdims", False):
                    # In this case the first arg is the initial tensor and
                    # the second arg is (optionally) the axis
                    tensor = args[0]
                    axis = bound_args.get("axis")
                    result = _util.apply_keepdims(result, axis, tensor.ndim)

            # out
            if "out" in params:
                # out can be in any position so we need sig.bind
                if bound_args is None:
                    bound_args = sig.bind(*args, **kwds).arguments
                out = bound_args.get("out")
                result = maybe_copy_to(out, result, promote_scalar_result)
            result = wrap_tensors(result)

            return result

        return wrapped

    if _func is None:
        return normalizer_inner
    else:
        return normalizer_inner(_func)
