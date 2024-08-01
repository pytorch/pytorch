# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union

import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch import sym_float, sym_int
from torch._prims_common import (
    BoolLike,
    DeviceLikeType,
    Dim,
    DimsSequenceType,
    DimsType,
    dtype_to_type,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    FloatLike,
    FloatWithoutSymFloat,
    IntLike,
    is_weakly_lesser_type,
    Number,
    NumberType,
    RealNumberType,
    REDUCTION_OUTPUT_TYPE_KIND,
    ShapeType,
    StrideType,
    TensorLike,
    TensorLikeType,
    TensorOrNumberLikeType,
    TensorSequenceType,
)
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    _maybe_resize_out,
    _safe_copy_out,
    elementwise_type_promotion_wrapper,
    elementwise_unary_scalar_wrapper,
    out_wrapper,
)


# Experimental module containing prototype Python references for existing
#   PyTorch operations.

__all__ = [
    #
    # Elementwise Unary References
    #
    "abs",
    "acos",
    "acosh",
    "asinh",
    "asin",
    "atan",
    "atanh",
    "bitwise_not",
    # "cbrt",  # No corresponding torch operation
    "ceil",
    "conj_physical",
    "cos",
    "cosh",
    "count_nonzero",
    "deg2rad",
    "digamma",
    "erf",
    "erfinv",
    "erfc",
    "exp",
    "expm1",
    "exponential",
    "exp2",
    "fill",
    "fill_",
    "floor",
    "frac",
    "geometric",
    "index_add",
    "index_copy",
    "index_copy_",
    "index_select",
    "index_fill",
    "index_fill_",
    "isfinite",
    "isinf",
    "isposinf",
    "isneginf",
    "isnan",
    "isreal",
    "i0",
    "lerp",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "log_normal",
    "log_softmax",
    "mvlgamma",
    "norm",
    "normal",
    "nan_to_num",
    "neg",
    "positive",
    "rad2deg",
    "reciprocal",
    "round",  # TODO: model kwargs
    "sigmoid",
    "sgn",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "softmax",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trace",
    "trunc",
    #
    # Elementwise Binary References
    #
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "clamp_min",
    "clamp_max",
    "copysign",
    "div",
    "eq",
    "float_power",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "gcd",
    "ge",
    "gt",
    "heaviside",
    "hypot",
    "igamma",
    "igammac",
    "imag",
    "isclose",
    "lcm",
    # 'ldexp',
    "le",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logsumexp",
    "lt",
    # 'max', # implement with reductions
    "maximum",
    # 'min', # implement with reductions
    "minimum",
    "mul",
    "ne",
    "nextafter",
    # 'polar',  # abs, cos, sin
    "pow",
    "real",
    "rpow",
    "remainder",
    "rsub",
    "rtruediv",
    "rfloordiv",
    "sub",
    "true_divide",
    "trunc_divide",
    "xlogy",
    #
    # Elementwise Ternary References
    #
    "addcdiv",
    "addcmul",
    "clamp",
    #
    # Conditional references
    #
    "masked_fill",
    "masked_fill_",
    "where",
    #
    # Data conversion and movement references
    #
    "clone",
    "copy_to",  # TODO: add OpInfo (or implement .to)
    "item",
    "to",
    #
    # Reduction ops
    #
    "all",
    "amax",
    "amin",
    "any",
    "cumsum",
    "cumprod",
    "mean",
    "dot",
    "vdot",
    "std",
    "std_mean",
    "sum",
    "sum_to_size",
    "prod",
    "var",
    "var_mean",
    #
    # Linear algebra ops
    #
    "addr",
    #
    # View & Shape Ops
    #
    "alias",
    "alias_copy",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "as_strided",
    "as_strided_copy",
    "as_strided_scatter",
    "block_diag",
    "broadcast_shapes",
    "broadcast_tensors",
    "broadcast_to",
    "cat",
    "chunk",
    "column_stack",
    "conj",
    "constant_pad_nd",
    "contiguous",
    "diag_embed",
    "diag",
    "diagonal",
    "diagonal_copy",
    "diagonal_scatter",
    "dsplit",
    "dstack",
    "expand",
    "expand_as",
    "expand_copy",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "hsplit",
    "hstack",
    "meshgrid",
    "movedim",
    "narrow",
    "narrow_copy",
    "native_group_norm",
    "native_layer_norm",
    "permute",
    "ravel",
    "repeat",
    "reshape",
    "reshape_as",
    "roll",
    "rot90",
    "rsqrt",
    "stack",
    "swap_axes",  # alias for transpose
    "squeeze",
    "t",
    "t_copy",
    "T",
    "take_along_dim",
    "tensor_split",
    "transpose",
    "unfold",
    "unfold_copy",
    "unsqueeze",
    "unsqueeze_copy",
    "view",
    "view_as",
    "view_copy",
    "vsplit",
    "vstack",
    "view_as_complex",
    "unflatten",
    "unbind",
    "triu",
    "tril",
    "triu_indices",
    "tril_indices",
    #
    # Tensor Creation
    #
    "arange",
    "cauchy",
    "empty",
    "empty_like",
    "empty_permuted",
    "empty_strided",
    "eye",
    "full",
    "full_like",
    "linspace",
    "logspace",
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_ones",
    "new_zeros",
    "ones",
    "ones_like",
    "randn",
    "scalar_tensor",
    "zero",
    "zeros",
    "zeros_like",
    #
    # Test-related functions
    #
    "allclose",
    "equal",
    #
    # Statistical operations
    #
    "bucketize",
    #
    # Misc
    #
    "is_complex",
    "renorm",
    "stft",
    "istft",
]

Tensor = torch.Tensor
DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]
aten = torch._ops.ops.aten

# Note that the docstrings for the public methods from this file are in
# torch/_torch_docs.py


def is_noncontiguous_supported(device):
    return device is None or device.type != "hpu"


def handle_noncontiguous_outputs(input_tlist, output):
    device = None
    from torch._subclasses.fake_tensor import FakeTensor

    for t in input_tlist:
        if isinstance(t, FakeTensor):
            device = t.fake_device
            break

    if not is_noncontiguous_supported(device):
        output = output.contiguous()

    return output


def _broadcast_shapes(*_shapes):
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    shapes = tuple(
        (x,) if isinstance(x, IntLike) else x
        for x in filter(lambda x: x is not None, _shapes)
    )

    # Short-circuits on no input
    if len(shapes) == 0:
        return None

    # Type checking
    # TODO: make common validations available as utils
    for shape in shapes:
        assert isinstance(shape, Sequence)

    # Computes common shape
    common_shape = [
        1,
    ] * reduce(max, (len(shape) for shape in shapes))
    for arg_idx, shape in enumerate(shapes):
        for idx in range(-1, -1 - len(shape), -1):
            if guard_size_oblivious(common_shape[idx] == 1):
                if shape[idx] < 0:
                    raise ValueError(
                        "Attempting to broadcast a dimension with negative length!"
                    )
                common_shape[idx] = shape[idx]
            elif guard_size_oblivious(shape[idx] != 1):
                if common_shape[idx] != shape[idx]:
                    raise RuntimeError(
                        f"Attempting to broadcast a dimension of length {shape[idx]} at {idx}! "
                        f"Mismatching argument at index {arg_idx} had {shape}; but expected shape "
                        f"should be broadcastable to {common_shape}"
                    )

    return common_shape


def _maybe_broadcast(*args, preserve_cpu_scalar_tensors=True):
    # Computes common shape
    common_shape = _broadcast_shapes(
        *(t.shape if isinstance(t, TensorLike) else None for t in args)
    )

    def __maybe_broadcast(x, shape):
        if x is None:
            return None
        elif isinstance(x, Number):
            return x
        elif isinstance(x, TensorLike):
            if preserve_cpu_scalar_tensors and utils.is_cpu_scalar_tensor(x):
                return x

            if not utils.same_shape(x.shape, common_shape):
                return x.expand(common_shape)

            return x
        else:
            raise RuntimeError(
                "Unexpected type when broadcasting: " + str(type(x)) + "!"
            )

    return tuple(__maybe_broadcast(x, common_shape) for x in args)


# Utilities should come BEFORE this import
from torch._decomp import register_decomposition


#
# Elementwise unary references
#

infer_aten_op = object()


# TODO: add type promotion support
def _make_elementwise_unary_reference(
    type_promotion_kind,
    *,
    aten_op=infer_aten_op,
    extra_meta=None,
) -> Callable:
    def inner(prim: Callable):
        nonlocal aten_op

        @wraps(prim)
        @out_wrapper()
        @elementwise_unary_scalar_wrapper
        @elementwise_type_promotion_wrapper(
            type_promoting_args=("a",),
            type_promotion_kind=type_promotion_kind,
        )
        def _ref(a: TensorLikeType) -> TensorLikeType:
            if extra_meta is not None:
                extra_meta(a)

            output = prim(a)
            return handle_noncontiguous_outputs([a], output)

        if aten_op is infer_aten_op:
            aten_op = utils.get_aten_op(prim, prim.__name__)
        if aten_op is not None:
            register_decomposition(aten_op)(_ref)

        return _ref

    return inner


def _make_alias(fn, name):
    """
    This function defines an alias of another function and sets its __name__ argument.
    It also sets its __module__ argument to the module of the caller.
    Note that when naively doing `alias = fn`, we have that `alias.__name__ == "fn"`, and
    `alias.__module__ == fn.__module__`.
    """

    def _fn(*args, **kwargs):
        return fn(*args, **kwargs)

    _fn.__name__ = name
    _fn.__module__ = inspect.currentframe().f_back.f_globals["__name__"]  # type: ignore[union-attr]
    return _fn


def _make_inplace(fn):
    """
    Given a function with out variant (i.e. using `out_wrapper()), it returns its in-place variant
    See https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-do-in-place-operations-work-in-pytorch
    """

    # nb. We use the name of the first argument used in the unary references
    @wraps(fn)
    def _fn(a, *args, **kwargs):
        return fn(a, *args, out=a, **kwargs)

    inplace_name = f"{fn.__name__}_"
    _fn.__name__ = inplace_name
    _fn = register_decomposition(getattr(aten, inplace_name))(_fn)  # type: ignore[assignment]

    # We access the __all__ attribute of the module where fn is defined
    # There may be a cleaner way of doing this...
    from inspect import getmodule

    _all = getmodule(fn).__all__  # type: ignore[union-attr]
    if inplace_name not in _all:
        _all.append(inplace_name)
    return _fn


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT)
def abs(a):
    return prims.abs(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def acos(a):
    return prims.acos(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def acosh(a):
    return prims.acosh(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def asin(a):
    return prims.asin(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def asinh(a):
    return prims.asinh(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def atan(a):
    return prims.atan(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def atanh(a):
    return prims.atanh(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def bitwise_not(a):
    return prims.bitwise_not(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def ceil(a):
    return prims.ceil(a)


@register_decomposition(aten.is_complex)
def is_complex(input: TensorLikeType):
    return utils.is_complex_dtype(input.dtype)


@register_decomposition(aten.conj_physical)
@out_wrapper()
def conj_physical(input: TensorLikeType):
    if not utils.is_complex_dtype(input.dtype):
        return input
    return prims.conj_physical(input)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def cos(a):
    return prims.cos(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def cosh(a):
    return prims.cosh(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def digamma(a):
    return prims.digamma(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def erf(a):
    return prims.erf(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def erfinv(a):
    return prims.erf_inv(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def erfc(a):
    return prims.erfc(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def exp(a):
    return prims.exp(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def expm1(a):
    return prims.expm1(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def exp2(a):
    return prims.exp2(a)


# Fill has its own implementation because it has a value parameter
# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a,"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def fill(a: TensorLikeType, value: NumberType) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    assert isinstance(value, Number)

    python_type = utils.dtype_to_type(a.dtype)
    if not utils.is_weakly_lesser_type(type(value), python_type):
        msg = f"value argument of type {type(value)} cannot be safely cast to type {python_type}!"
        raise ValueError(msg)

    return prims.fill(a, value)


def fill_(a: TensorLikeType, value: NumberType) -> TensorLikeType:
    r = prims.fill(a, value)
    prims.copy_to(a, r)
    return a


@register_decomposition(aten.zero)
@out_wrapper()
def zero(input: TensorLikeType) -> TensorLikeType:
    return torch.zeros_like(input)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def floor(a):
    return prims.floor(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def frac(x: TensorLikeType) -> TensorLikeType:
    trunc_x = torch.mul(torch.floor(torch.abs(x)), torch.sign(x))
    return torch.sub(x, trunc_x)


# imag does not use _make_elementwise_unary_reference because it does not support out
def imag(a: TensorLikeType) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    torch._check(
        utils.is_complex_dtype(a.dtype), lambda: "imag only supports complex tensors."
    )
    return prims.imag(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=None,  # CompositeImplicitAutograd
)
def isfinite(a: TensorLikeType) -> TensorLikeType:
    if utils.is_float_dtype(a.dtype) or utils.is_complex_dtype(a.dtype):
        return prims.isfinite(a)

    return ones_like(a, dtype=torch.bool)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def isinf(a: TensorLikeType) -> TensorLikeType:
    if utils.is_complex_dtype(a.dtype):
        return torch.logical_or(isinf(torch.real(a)), isinf(torch.imag(a)))
    if utils.is_float_dtype(a.dtype):
        return torch.abs(a) == float("inf")
    return torch.zeros_like(a, dtype=torch.bool)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def isposinf(a: TensorLikeType) -> TensorLikeType:
    torch._check(
        not utils.is_complex_dtype(a.dtype),
        lambda: f"Complex dtype is not supported for isposinf, got dtype {a.dtype}",
    )
    if utils.is_float_dtype(a.dtype):
        return a == float("inf")
    return torch.zeros_like(a, dtype=torch.bool)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def isneginf(a: TensorLikeType) -> TensorLikeType:
    torch._check(
        not utils.is_complex_dtype(a.dtype),
        lambda: f"Complex dtype is not supported for isneginf, got dtype {a.dtype}",
    )
    if utils.is_float_dtype(a.dtype):
        return a == float("-inf")
    return torch.zeros_like(a, dtype=torch.bool)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def isnan(a: TensorLikeType) -> TensorLikeType:
    return prims.ne(a, a)


# alias
mvlgamma = _make_alias(torch.special.multigammaln, "mvlgamma")  # type: ignore[has-type]


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=None,  # CompositeImplicitAutograd
)
def isreal(a: TensorLikeType) -> TensorLikeType:
    if utils.is_complex_dtype(a.dtype):
        return torch.imag(a) == 0
    return torch.ones_like(a, dtype=torch.bool)


# TODO: if this is special maybe it should be defined there and imported here?
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, aten_op=aten.i0
)
def i0(a):
    return prims.bessel_i0(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def lgamma(a):
    return prims.lgamma(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def log(a):
    return prims.log(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def log1p(a):
    return prims.log1p(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def log2(a):
    return prims.log2(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def log10(a):
    return prims.log10(a)


# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
def log_softmax(
    a: TensorLikeType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    result_dtype = dtype or a.dtype
    computation_dtype = utils.get_computation_dtype(result_dtype)
    a_ = _maybe_convert_to_dtype(a, computation_dtype)
    return _maybe_convert_to_dtype(a_ - logsumexp(a_, dim, keepdim=True), result_dtype)  # type: ignore[return-value]


@register_decomposition(aten.logsumexp)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def logsumexp(
    self: TensorLikeType, dim: DimsType, keepdim: bool = False
) -> TensorLikeType:
    if not isinstance(dim, Iterable):
        dim = (dim,)
    if self.numel() == 0:
        return torch.sum(torch.exp(self), dim, keepdim).log()
    maxes = torch.amax(self, dim, keepdim=True)
    maxes = torch.masked_fill(maxes, maxes.abs() == float("inf"), 0)
    maxes_squeezed = maxes if keepdim else torch.squeeze(maxes, dim)
    result = torch.sum(torch.exp(self - maxes), dim, keepdim)
    return result.log().add(maxes_squeezed)


@register_decomposition(aten.nan_to_num)
@out_wrapper()
def nan_to_num(
    a: TensorLikeType,
    nan: Optional[NumberType] = 0.0,
    posinf: Optional[NumberType] = None,
    neginf: Optional[NumberType] = None,
) -> TensorLikeType:
    assert isinstance(a, TensorLike)

    if utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
        return a.clone()

    if nan is None:
        nan = 0.0

    if posinf is None:
        posinf = torch.finfo(a.dtype).max

    if neginf is None:
        neginf = torch.finfo(a.dtype).min

    result = torch.where(torch.isnan(a), nan, a)  # type: ignore[call-overload]
    result = torch.where(torch.isneginf(a), neginf, result)  # type: ignore[call-overload]
    result = torch.where(torch.isposinf(a), posinf, result)  # type: ignore[call-overload]
    return result


def _neg_meta(a: TensorLikeType):
    torch._check(
        a.dtype is not torch.bool,
        lambda: (
            "Negation, the `-` operator, on a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or `logical_not()` "
            "operator instead."
        ),
    )


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, extra_meta=_neg_meta
)
def neg(a):
    return prims.neg(a)


# positive does not use _make_elementwise_unary_reference because it does not support out
# CompositeImplicitAutograd - don't register decomp
def positive(a: TensorLikeType) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    if a.dtype is torch.bool:
        msg = "positive does not support bool tensors."
        raise RuntimeError(msg)
    return a


# real does not use _make_elementwise_unary_reference because it does not support out
def real(a: TensorLikeType) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    if utils.is_complex_dtype(a.dtype):
        return prims.real(a)
    return a


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def reciprocal(a):
    return prims.reciprocal(a)


@register_decomposition(aten.round)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def round(a: TensorLikeType, *, decimals: int = 0) -> TensorLikeType:
    if decimals == 0:
        return prims.round(a)
    else:
        ten_pow = 10**decimals
        ten_neg_pow = 10 ** (-decimals)
        return prims.mul(prims.round(prims.mul(a, ten_pow)), ten_neg_pow)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def rsqrt(a):
    return prims.rsqrt(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sigmoid(a: TensorLikeType) -> TensorLikeType:
    return true_divide(1, add(1, exp(neg(a))))


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def sgn(a):
    if utils.is_complex_dtype(a.dtype):
        a_abs = a.abs()
        return torch.where(a_abs == 0, 0, a / a_abs)
    else:
        return a.sign()


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def sign(a):
    return prims.sign(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def signbit(a):
    return prims.signbit(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sin(a):
    return prims.sin(a)


# Autograd note: This will give the right first derivative at zero (by chance),
# but not the right second derivative
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sinc(a):
    a = math.pi * a
    return torch.where(a == 0, 1, torch.sin(a) / a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sinh(a):
    return prims.sinh(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sqrt(a):
    return prims.sqrt(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG,
    aten_op=None,  # CompositeImplicitAutograd,
)
def square(a: TensorLikeType) -> TensorLikeType:
    return mul(a, a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def tan(a):
    return prims.tan(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def tanh(a):
    return prims.tanh(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def trunc(a):
    return prims.trunc(a)


# TODO: register this as a real ref/decomposition once TorchInductor supports complex!
def view_as_complex(self: TensorLikeType) -> TensorLikeType:
    input_dtype = self.dtype
    torch._check(
        utils.is_float_dtype(input_dtype),
        lambda: f"view_as_complex is only supported for floating point"
        f"tensors, but got a tensor of scalar type: {input_dtype}",
    )
    sizes = self.size()
    torch._check(
        len(sizes) != 0,
        lambda: "Input tensor must have one or more dimensions",
    )
    torch._check(
        sizes[-1] == 2,
        lambda: "Tensor must have a last dimension of size 2",
    )

    old_strides = self.stride()
    torch._check(
        old_strides[-1] == 1,
        lambda: "Tensor must have a last dimension with stride 1",
    )
    dims = old_strides[:-1]
    torch._check(
        builtins.all(stride % 2 == 0 for stride in dims),
        lambda: "Tensor must have a stride divisible by 2 for all but last dimension",
    )
    torch._check(
        self.storage_offset() % 2 == 0,
        lambda: "Tensor must have a storage_offset divisible by 2",
    )
    return prims.view_element_type(
        self, utils.corresponding_complex_dtype(input_dtype)
    ).squeeze(-1)


def _make_elementwise_binary_reference(
    type_promotion_kind,
    aten_op=infer_aten_op,
    name=None,
    has_out=True,
    supports_lhs_python_scalar=True,
    supports_rhs_python_scalar=True,
    supports_two_python_scalars=False,
    should_register_decomposition=True,
) -> Callable:
    def inner(prim: Callable):
        nonlocal aten_op, name
        if name is None:
            name = prim.__name__

        @wraps(prim)
        @elementwise_type_promotion_wrapper(
            type_promoting_args=("a", "b"),
            type_promotion_kind=type_promotion_kind,
        )
        def _ref(
            a: Union[Tensor, NumberType],
            b: Union[Tensor, NumberType],
        ) -> Tensor:
            torch._check_value(
                supports_lhs_python_scalar or not isinstance(a, Number),
                lambda: f"{name}: Received a lhs Python scalar to an elementwise binary "
                "operation that does not accept lhs scalars!",
            )
            torch._check_value(
                supports_rhs_python_scalar or not isinstance(b, Number),
                lambda: f"{name}: Received a rhs Python scalar to an elementwise binary "
                "operation that does not accept rhs scalars!",
            )
            torch._check_value(
                supports_two_python_scalars
                or not (isinstance(a, Number) and isinstance(b, Number)),
                lambda: f"{name}: Receive two Number inputs to an elementwise binary operation!",
            )
            a, b = _maybe_broadcast(a, b)
            output = prim(a, b)
            return handle_noncontiguous_outputs([a, b], output)

        if has_out:
            _ref = out_wrapper()(_ref)  # type: ignore[assignment]

        _ref.__name__ = name
        if aten_op is infer_aten_op:
            aten_op = utils.get_aten_op(prim, name)
        if aten_op is not None and should_register_decomposition:
            register_decomposition(aten_op)(_ref)

        return _ref

    return inner


# Add has its own implementation because it has an alpha argument
@register_decomposition(aten.add)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def add(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    alpha: Optional[NumberType] = None,
):
    """
    Reference implementation of torch.add
    """

    a, b = _maybe_broadcast(a, b)

    if alpha is not None:
        dtype = a.dtype if isinstance(a, TensorLike) else b.dtype  # type: ignore[union-attr]
        python_type = utils.dtype_to_type(dtype)
        if python_type != bool and not utils.is_weakly_lesser_type(
            type(alpha), python_type
        ):
            msg = f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!"
            raise ValueError(msg)
        if isinstance(b, TensorLike):
            b = prims.mul(b, alpha)
        else:
            b = b * alpha

    output = prims.add(a, b)
    return handle_noncontiguous_outputs([a, b], output)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def atan2(a, b):
    return prims.atan2(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_and(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.bitwise_and(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_left_shift(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.shift_left(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_or(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.bitwise_or(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_right_shift(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.shift_right_arithmetic(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_xor(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.bitwise_xor(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
)
def copysign(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    if isinstance(b, Number) and isinstance(a, Tensor):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(a, Tensor) and isinstance(b, Tensor) and a.device != b.device:
        msg = f"Expected divisor (b) to be on the same device ({a.device}) as dividend (a), but it is found on {b.device}!"
        raise RuntimeError(msg)
    return where(signbit(b), neg(abs(a)), abs(a))


# complex =  _make_elementwise_binary_reference(prims.complex, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)


@register_decomposition(aten.div)
@out_wrapper()
def div(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    rounding_mode: Optional[str] = None,
):
    """
    Reference implementation of torch.div
    """
    if rounding_mode is None:
        return true_divide(a, b)
    elif rounding_mode == "trunc":
        return trunc_divide(a, b)
    elif rounding_mode == "floor":
        return floor_divide(a, b)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def eq(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.eq(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG,
)
def pow(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
) -> TensorLikeType:
    assert isinstance(a, TensorLikeType) or isinstance(b, TensorLikeType)

    if isinstance(b, Number):
        if b == 1.0:
            return a.clone()  # type: ignore[return-value,union-attr]
        elif b == 2.0:
            return a * a  # type: ignore[return-value]
        elif b == 0.5:
            return torch.sqrt(a)  # type: ignore[arg-type]
    elif isinstance(a, Number):
        if a == 1.0:
            return torch.fill(b, True)
        if a == 2.0 and (
            utils.is_float_dtype(b.dtype) or utils.is_complex_dtype(b.dtype)
        ):
            return torch.exp2(b)

    return prims.pow(a, b)


# Float power has its own implementation because it has unique type promotion.
# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
def float_power(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
) -> Tensor:
    if isinstance(a, Number) and isinstance(b, Number):
        raise ValueError(
            "Receive two Number inputs to an elementwise binary operation!"
        )

    # Handles type promotion
    dtype = utils.get_higher_dtype(a, b)
    assert dtype is not None
    if utils.is_complex_dtype(dtype):
        dtype = torch.complex128
    else:
        dtype = torch.float64

    # Float power has the following contiguous cast behavior to be
    # consistent with its C++ impl
    a = _maybe_convert_to_dtype(a, dtype)
    b = _maybe_convert_to_dtype(b, dtype)

    a, b = _maybe_broadcast(a, b)
    return pow(a, b)


# >>> a = torch.tensor(-0.2500, dtype=torch.float64)
# tensor(-0.250000000000000, dtype=torch.float64)
#
# >>> b = torch.tensor(-0.0010, dtype=torch.float64)
# tensor(-0.001000000000000, dtype=torch.float64)
#
# Note: In this case, casting float to double will expand the float mantissa with zeros,
# while creating a double generates a distinct mantissa.
# >>> torch.tensor(-0.001).to(dtype=torch.float64)
# tensor(-0.001000000047497, dtype=torch.float64)
#
# Floor Division
# The difference is caused because torch.remainder(a, b) = -0.001.
#
# >>> torch.floor(torch.true_divide(a, b))
# tensor(250., dtype=torch.float64)
#
# >>> torch.div(a, b, rounding_mode='floor')
# tensor(249., dtype=torch.float64)
#
# Definition: a // b = (a - remainder(a, b)) / b
# >>> torch.true_divide(torch.sub(a, torch.remainder(a, b)), b)
# tensor(249., dtype=torch.float64)
#
# For reference, see CPython's implementation:
# https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636


@_make_elementwise_binary_reference(
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_two_python_scalars=True,
    should_register_decomposition=False,
)
def floor_divide(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    # Wrap scalars because some references only accept tensor arguments.
    if isinstance(a, Number) and isinstance(b, Number):
        a = scalar_tensor(a)
        b = scalar_tensor(b)
    elif isinstance(b, Number) and isinstance(a, Tensor):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(a, Number) and isinstance(b, Tensor):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)
    elif isinstance(a, Tensor) and isinstance(b, Tensor) and a.device != b.device:
        if a.device == torch.device("cpu"):
            msg = f"Expected divisor (b) to be on the same device ({a.device}) as dividend (a), but it is found on {b.device}!"
            raise RuntimeError(msg)
        else:
            b = prims.device_put(b, device=a.device)

    assert isinstance(a, Tensor) and isinstance(b, Tensor)
    dtype = a.dtype
    if utils.is_float_dtype(dtype):
        return _floor_divide_float(a, b)
    elif utils.is_integer_dtype(dtype):
        return _floor_divide_integer(a, b)
    else:
        torch._check(False, lambda: f"{dtype} not supported for floor_divide")


def _floor_divide_integer(a: Tensor, b: Tensor) -> Tensor:
    a, b = _maybe_broadcast(a, b)

    if not a.dtype.is_signed:
        return prims.div(a, b)

    # Convert truncation to flooring:
    offset = (torch.signbit(a) != torch.signbit(b)).logical_and(torch.fmod(a, b) != 0)
    return prims.div(a, b) - _maybe_convert_to_dtype(offset, a.dtype)


def _floor_divide_float(a: Tensor, b: Tensor) -> Tensor:
    mod = fmod(a, b)
    div = true_divide(sub(a, mod), b)

    # Ensure that the remainder has the same sign as denominator
    different_signed_inputs = bitwise_xor(lt(a, 0), lt(b, 0))
    non_zero_remainder = ne(mod, 0)
    mask = bitwise_and(non_zero_remainder, different_signed_inputs)
    div = where(mask, sub(div, 1), div)

    # Map quotient to nearest integer value
    floor_div = floor(div)
    mask = gt(sub(div, floor_div), 0.5)
    floor_div = where(mask, add(floor_div, 1), floor_div)

    basic_div = true_divide(a, b)
    zero_tensor = scalar_tensor(0, dtype=basic_div.dtype, device=basic_div.device)

    # If quotient is zero, copy signbit from true_divide quotient
    floor_div = where(ne(div, 0), floor_div, copysign(zero_tensor, basic_div))

    # If denominator is zero, then follow true_divide behavior
    return where(ne(b, 0), floor_div, basic_div)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def fmax(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.fmax(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def fmin(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.fmin(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=True,
)
def fmod(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.fmod(a, b)


@register_decomposition(aten.frexp)
@out_wrapper("mantissa", "exponent")
def frexp(self: TensorLikeType) -> Tuple[TensorLikeType, TensorLikeType]:
    return torch.return_types.frexp(prims.frexp(self))


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def gcd(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.gcd(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def ge(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.ge(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def gt(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.gt(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def heaviside(input: TensorLikeType, values: TensorLikeType) -> TensorLikeType:
    input_eq_zero = torch.eq(input, 0)
    input_lt_zero = torch.logical_or(torch.lt(input, 0), torch.isnan(input))
    zeros_and_ones = torch.where(input_lt_zero, 0, 1)
    output = torch.where(input_eq_zero, values, zeros_and_ones)
    return output


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def hypot(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.hypot(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def igamma(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.igamma(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def igammac(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.igammac(a, b)


def _check_close_args(
    name: str,
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float,
    atol: float,
) -> None:
    torch._check_value(
        a.dtype == b.dtype,
        lambda: f"{name}: Attempting to compare tensors of different dtypes {a.dtype} and {b.dtype}!",
    )
    torch._check(
        rtol >= 0,
        lambda: f"{name}: rtol must be greater than or equal to zero, but got {rtol}!",
    )
    torch._check(
        atol >= 0,
        lambda: f"{name}: atol must be greater than or equal to zero, but got {atol}!",
    )


# CompositeImplicitAutograd - don't register decomp
def isclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> TensorLikeType:
    _check_close_args(name="torch.isclose", a=a, b=b, rtol=rtol, atol=atol)

    close = eq(a, b)
    if equal_nan and (utils.is_float_dtype(a.dtype) or utils.is_complex_dtype(a.dtype)):
        close = logical_or(close, logical_and(isnan(a), isnan(b)))

    # Note: In case of zero tolerances the closeness inequality degenerates to an equality check.
    # In this case, the short-circuit prevents false positives as detailed in the paragraph below.
    if atol == 0 and rtol == 0:
        return close

    # Note [closeness error computation]
    # atol and rtol are provided as doubles, so the computation
    # rtol * other will produce a float or complex tensor.
    # When the difference (self - other) is compared to it then the
    # tensor representing the difference will also be cast to float or complex.
    # However, since (self - other) in uint8 is very likely to produce a
    # negative value, this moves the cast forward so the difference is
    # always computed in a float or complex type.
    # If the values of the integer tensors cannot be exactly represented
    # by the default scalar type then this may cause an incorrect result.
    if not utils.is_float_dtype(a.dtype) and not utils.is_complex_dtype(a.dtype):
        a = prims.convert_element_type(a, torch.get_default_dtype())
        b = prims.convert_element_type(b, torch.get_default_dtype())

    allowed_error = add(atol, abs(mul(b, rtol)))
    actual_error = abs(sub(a, b))

    # Computes finite closeness
    result = logical_or(
        close, logical_and(isfinite(actual_error), le(actual_error, allowed_error))
    )

    return result


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def lcm(a: TensorLikeType, b: TensorLikeType):
    dtype = a.dtype
    # promoting to int32 to maintain 100% consistency with C++ and to
    # prevent overflow in case of int8 and int16
    promote_to_int = dtype in (torch.int8, torch.int16)
    if promote_to_int:
        a = prims.convert_element_type(a, torch.int32)
        b = prims.convert_element_type(b, torch.int32)

    g = torch.gcd(a, b)
    # Avoid division by zero in case gcd(0, 0) == 0
    g = torch.where(g == 0, 1, g)
    res = torch.abs(prims.div(a, g) * b)
    return res if not promote_to_int else prims.convert_element_type(res, dtype)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def le(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.le(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def logaddexp(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # Nb. this implementation does not distribute the gradients evenly when a == b
    mask = torch.real(a) >= torch.real(b)
    max_ = torch.where(mask, a, b)
    min_ = torch.where(mask, b, a)
    inf_mask = torch.logical_and(
        torch.logical_not(torch.isfinite(torch.real(a))), torch.real(a) == torch.real(b)
    )
    if utils.is_complex_dtype(a.dtype) or utils.is_complex_dtype(b.dtype):
        # are you wondering what this bunch of codes are for? edge cases!
        neg_min_mask = torch.real(min_) < 0
        inf_vals = torch.where(
            neg_min_mask, min_, torch.log(torch.exp(min_) + torch.exp(max_))
        )
        non_nan_vals = torch.where(
            inf_mask, inf_vals, max_ + torch.log1p(torch.exp(min_ - max_))
        )
        # the type for full_like does not include tensor yet
        nan_mask = torch.isnan(min_)
        return torch.where(nan_mask, complex(float("nan"), float("nan")), non_nan_vals)  # type: ignore[call-overload]
    else:
        return torch.where(inf_mask, a, max_ + torch.log1p(torch.exp(min_ - max_)))


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def logaddexp2(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    torch._check(
        not (utils.is_complex_dtype(a.dtype) or utils.is_complex_dtype(b.dtype)),
        lambda: "logaddexp2 doesn't support complex dtypes",
    )
    # Nb. this implementation does not distribute the gradients evenly when a == b
    mask = a >= b
    max_ = torch.where(mask, a, b)
    min_ = torch.where(mask, b, a)
    inf_mask = torch.logical_and(torch.isinf(a), a == b)
    inv_log_2 = 1.0 / math.log(2)
    result = max_ + torch.log1p(torch.exp2(min_ - max_)) * inv_log_2
    return torch.where(inf_mask, a, result)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)
def logical_and(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        a = a != 0
    if not utils.is_boolean_dtype(b.dtype):
        b = b != 0
    return a & b


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def logical_not(a: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        return a == 0
    return ~a


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)
def logical_or(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        a = a != 0
    if not utils.is_boolean_dtype(b.dtype):
        b = b != 0
    return bitwise_or(a, b)


# TODO: skip unnecessary conversion of long to float
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)
def logical_xor(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        a = a != 0
    if not utils.is_boolean_dtype(b.dtype):
        b = b != 0
    return a ^ b


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def lt(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.lt(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def maximum(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.maximum(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def minimum(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.minimum(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_two_python_scalars=True,
)
def mul(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.mul(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def ne(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.ne(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def nextafter(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.nextafter(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def remainder(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.remainder(a, b)


# reverse sub
@register_decomposition(aten.rsub)
@out_wrapper()
def rsub(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    alpha: NumberType = 1,
):
    if isinstance(a, Number):
        msg = "Received a Number for the first argument, but expected a Tensor"
        raise ValueError(msg)

    return torch.sub(b, a, alpha=alpha)


# TODO: consider refactoring this with add impl
# sub has its own implementation because it has an alpha argument
@register_decomposition(aten.sub)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def sub(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    alpha: NumberType = 1,
):
    """
    Reference implementation of torch.sub
    """

    a, b = _maybe_broadcast(a, b)

    if isinstance(a, TensorLike) and isinstance(b, TensorLike):
        torch._check(
            not utils.is_boolean_dtype(a.dtype) and not utils.is_boolean_dtype(b.dtype),
            lambda: (
                "Subtraction, the `-` operator, with two bool tensors is not supported. "
                "Use the `^` or `logical_xor()` operator instead."
            ),
        )

    if alpha != 1:
        dtype = a.dtype if isinstance(a, TensorLike) else b.dtype  # type: ignore[union-attr]
        python_type = utils.dtype_to_type(dtype)
        if not utils.is_weakly_lesser_type(type(alpha), python_type):
            msg = f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!"
            raise ValueError(msg)
        if isinstance(b, torch.Tensor):
            b = prims.mul(b, alpha)
        else:
            # Carefully not to use prims.mul if b is a scalar / symint.
            # prims.mul always returns a tensor,
            # which will mess with type promotion.
            b = b * alpha

    output = prims.sub(a, b)
    return handle_noncontiguous_outputs([a, b], output)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    name="true_divide",
    aten_op=None,  # CompositeImplicitAutograd
    supports_two_python_scalars=True,
)
def true_divide(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.div(a, b)


@register_decomposition(aten.xlogy)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def xlogy(a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]):
    torch._check(
        isinstance(a, TensorLike) or isinstance(b, TensorLike),
        lambda: 'Expected either argument a or b to be a Tensor"',
    )

    # Operations like eq and log do not handle scalar values, so we convert them to scalar_tensors.
    if isinstance(b, TensorLike) and isinstance(a, Number):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)
    elif isinstance(a, TensorLike) and isinstance(b, Number):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)

    # mypy: expected "Tensor"
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)
    rhs = torch.where(torch.eq(a, 0), 0, torch.mul(a, torch.log(b)))
    return torch.where(torch.isnan(b), float("nan"), rhs)


@_make_elementwise_binary_reference(
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=None,  # CompositeImplicitAutograd
    supports_two_python_scalars=True,
)
def trunc_divide(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    dtype = utils.get_dtype(a)
    if utils.is_integer_dtype(dtype):
        return prims.div(a, b)

    return trunc(prims.div(a, b))


#
# Elementwise Ternary References
#


@register_decomposition(aten.addcdiv)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "tensor1", "tensor2"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def addcdiv(
    self: TensorLikeType,
    tensor1: TensorLikeType,
    tensor2: TensorLikeType,
    *,
    value: NumberType = 1,
) -> TensorLikeType:
    """
    Reference implementation of torch.addcdiv
    """
    if value is not None:
        dtype = self.dtype  # no scalars allowed, see add
        python_type = utils.dtype_to_type(dtype)
        torch._check_value(
            utils.is_weakly_lesser_type(type(value), python_type),
            lambda: f"value argument of type {type(value)} cannot be safely cast to type {python_type}!",
        )

    return self + value * tensor1 / tensor2


@register_decomposition(aten.addcmul)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "tensor1", "tensor2"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def addcmul(
    self: TensorLikeType,
    tensor1: TensorLikeType,
    tensor2: TensorLikeType,
    *,
    value: NumberType = 1,
) -> TensorLikeType:
    """
    Reference implementation of torch.addcmul
    """
    if value is not None:
        dtype = self.dtype  # no scalars allowed, see add
        python_type = utils.dtype_to_type(dtype)
        torch._check_value(
            utils.is_weakly_lesser_type(type(value), python_type),
            lambda: f"value argument of type {type(value)} cannot be safely cast to type {python_type}!",
        )

    return self + value * tensor1 * tensor2


@register_decomposition(aten.clamp)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "min", "max"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def clamp(
    a: TensorLikeType,
    min: Optional[TensorOrNumberLikeType] = None,
    max: Optional[TensorOrNumberLikeType] = None,
) -> TensorLikeType:
    # NOTE: grad behavior with implementation `where` is not consistent on `nan`
    if min is None and max is None:
        msg = "clamp called but both min and max are none!"
        raise ValueError(msg)
    if min is not None:
        a_isnan = torch.isnan(a)
        condition = torch.bitwise_or(torch.ge(a, min), a_isnan)  # type: ignore[arg-type]
        # we should also propagate `nan` coming from boundaries. However, that's
        # not necessary since `ge` would already `False` when either operands has
        # a `nan`. So this line below is redundant
        #   `condition = bitwise_and(condition, bitwise_not(isnan(min)))`
        a = torch.where(condition, a, min)  # type: ignore[arg-type]
    if max is not None:
        a_isnan = torch.isnan(a)
        # same as above, no need to adjust `nan` from `max`
        condition = torch.bitwise_or(torch.le(a, max), a_isnan)  # type: ignore[arg-type]
        a = torch.where(condition, a, max)  # type: ignore[arg-type]

    return a


@register_decomposition(aten.clamp_min)
@out_wrapper()
def clamp_min(
    self: TensorLikeType,
    min: Optional[TensorOrNumberLikeType] = None,
) -> TensorLikeType:
    return torch.clamp(self, min=min)  # type: ignore[arg-type]


@register_decomposition(aten.clamp_max)
@out_wrapper()
def clamp_max(
    self: TensorLikeType,
    max: Optional[TensorOrNumberLikeType] = None,
) -> TensorLikeType:
    return torch.clamp(self, max=max)  # type: ignore[arg-type]


#
# Conditional references
#


# https://pytorch.org/docs/stable/generated/torch.where.html
# TODO: implement alternate where
@register_decomposition(aten.where)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def where(
    pred: Tensor,
    a: Optional[TensorOrNumberLikeType] = None,
    b: Optional[TensorOrNumberLikeType] = None,
):
    """ """

    if a is None or b is None:
        raise NotImplementedError

    utils.check_same_device(pred, a, b, allow_cpu_scalar_tensors=True)
    torch._check(
        pred.dtype is torch.bool,
        lambda: f"expected predicate to be bool, got {pred.dtype}",
    )

    pred, a, b = _maybe_broadcast(pred, a, b)
    return prims.where(pred, a, b)


#
# Data Movement References
#
@register_decomposition(aten.clone)
@out_wrapper()
def clone(
    a: TensorLikeType, *, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    result = prims.clone(a, memory_format=memory_format)
    return result


def copy_to(a: Tensor, b: Tensor, *, allow_cross_device=True):
    if not allow_cross_device and a.device != b.device:
        msg = f"Attempting to copy from device {b.device} to device {a.device}, but cross-device copies are not allowed!"
        raise RuntimeError(msg)

    return prims.copy_to(a, b)


@register_decomposition(aten.item)
def item(a: TensorLikeType) -> NumberType:
    if a.numel() != 1:
        msg = f"Can't convert a tensor with {a.numel()} elements to a number!"
        raise ValueError(msg)

    # NOTE: explicit conversion is necessary for bool!
    # See https://github.com/pytorch/pytorch/issues/78071
    number_type = utils.dtype_to_type(a.dtype)
    return number_type(prims.item(a))


# fast path when `to` returns an alias to input. This mimics the same function in aten
def _to_will_alias(
    a: TensorLikeType,
    device: Optional[DeviceLikeType] = None,
    dtype: Optional[torch.dtype] = None,
    copy: Optional[bool] = None,
    layout: Optional[torch.layout] = None,
    memory_format: Optional[torch.memory_format] = None,
    pin_memory: Optional[bool] = False,
    non_blocking: bool = False,  # not using non_blocking
) -> bool:
    return (
        not copy
        and (device is None or a.device == device)
        and (dtype is None or a.dtype == dtype)
        and (layout is None or a.layout == layout)
        # is_pinned issue #84925
        # and (pin_memory is None or pin_memory == a.is_pinned())
        and (
            memory_format is None
            or memory_format == torch.preserve_format
            or utils.is_contiguous_for_memory_format(a, memory_format=memory_format)
        )
    )


@singledispatch
def _to_dispatch(*args, **kwargs):
    raise NotImplementedError


@_to_dispatch.register
def _to_device(
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> Dict[str, Any]:
    kwargs = {
        "device": device,
        "dtype": dtype,
        "non_blocking": non_blocking,
        "copy": copy,
        "memory_format": memory_format,
    }
    return kwargs


@_to_dispatch.register
def _to_device_str(
    device: str,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> Dict[str, Any]:
    kwargs = {
        "device": torch.device(device),
        "dtype": dtype,
        "non_blocking": non_blocking,
        "copy": copy,
        "memory_format": memory_format,
    }
    return kwargs


@_to_dispatch.register
def _to_dtype(
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> Dict[str, Any]:
    kwargs = {
        "dtype": dtype,
        "non_blocking": non_blocking,
        "copy": copy,
        "memory_format": memory_format,
    }
    return kwargs


@_to_dispatch.register
def _to_other(
    other: Tensor,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> Dict[str, Any]:
    device = other.device
    dtype = other.dtype
    layout = other.layout
    # is_pinned issue #84925
    # pin_memory = other.is_pinned()
    kwargs = {
        "device": device,
        "dtype": dtype,
        "layout": layout,
        "non_blocking": non_blocking,
        "copy": copy,
        "memory_format": memory_format,
    }
    return kwargs


# remove to_kwargs that is already present in `a`
def _canonicalize_to_arguments(a: Tensor, to_kwargs: dict):
    options_to_check = ["dtype", "device", "layout", "memory_format"]
    # "device" option could be passed a str instead torch.device
    if "device" in to_kwargs and isinstance(to_kwargs["device"], str):
        to_kwargs["device"] = torch.device(to_kwargs["device"])

    for kw in options_to_check:
        if kw in to_kwargs:
            if (
                (kw == "memory_format" and to_kwargs[kw] is torch.preserve_format)
                or (
                    kw == "device"
                    and to_kwargs[kw].type == a.device.type
                    and (
                        not to_kwargs[kw].index or to_kwargs[kw].index == a.device.index
                    )
                )
                or (
                    getattr(a, kw, None) == to_kwargs[kw]
                )  # this also handles {"memory_format": None}
            ):
                to_kwargs.pop(kw)


def to(a: TensorLikeType, *args, **kwargs) -> TensorLikeType:
    # handled dispatch via positional arguments
    if len(args) != 0:
        kwargs = _to_dispatch(*args, **kwargs)

    # TODO: is_pinned is not currently supported in refs or fake_tensor
    # https://github.com/pytorch/pytorch/issues/84925
    assert "pin_memory" not in kwargs
    _canonicalize_to_arguments(a, kwargs)

    if _to_will_alias(a, **kwargs):
        return a

    copy = kwargs.pop("copy") if "copy" in kwargs else False
    non_blocking = kwargs.pop("non_blocking") if "non_blocking" in kwargs else False

    # short-circuit to `prims.convert_element_type` when `to` is just a dtype change
    if (
        (copy or (kwargs.get("dtype", a.dtype) != a.dtype))
        and (not non_blocking)
        and ("memory_format" not in kwargs)
        and ("device" not in kwargs)
        and ("layout" not in kwargs)
        # is_pinned issue #84925
        # and ("pin_memory" not in kwargs)
    ):
        return prims.convert_element_type(a, kwargs.get("dtype", a.dtype))

    result = torch.empty_like(a, **kwargs)
    # TODO: non_blocking should be handled by `copy_to`
    copy_to(result, a)
    return result


#
# Reduction references
#


def _reduction(
    a: TensorLikeType,
    prim: Callable,
    *,
    has_identity: bool = True,
    accepts_dim_tuple: bool = True,  # to handle min/argmin that accept single dim only
    dims: Optional[DimsType] = None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,  # should be specified for ops that support it
    out: Optional[Tensor] = None,
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
) -> TensorLikeType:  # it is usually SAME, but I want
    # ref writers to actually think about what to put here
    assert isinstance(a, TensorLike)
    if a.ndim > 64:
        raise RuntimeError(
            f"Received a tensor with {a.ndim} dimensions, but only tensors with up to 64 dims are supported!"
        )

    if out is not None:
        assert isinstance(out, TensorLike)
        if dtype is not None:
            # TODO - this is true for eager mode currently, but it's wrong behavior for complex norms
            if dtype != out.dtype:
                raise RuntimeError(
                    "dtype argument and out dtype must match in reduction"
                )
    if not accepts_dim_tuple:
        assert dims is None or isinstance(dims, Dim)
    if isinstance(dims, Dim):
        dims = (dims,)  # type: ignore[assignment]
    dims = utils.reduction_dims(a.shape, dims)
    if not has_identity:
        valid_shape = a.ndim == 0 or builtins.all(a.shape[i] for i in dims)
        if not valid_shape:
            raise RuntimeError(
                "reducing over zero-size dimension for reduction operation without identity"
            )
    computation_dtype, result_dtype = utils.reduction_dtypes(
        a, output_dtype_kind, dtype
    )
    a = _maybe_convert_to_dtype(a, computation_dtype)  # type: ignore[method-assign]
    result = prim(a, dims)
    if keepdims:
        output_shape = [a.shape[i] if i not in dims else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dims]
        result = prims.broadcast_in_dim(result, output_shape, broadcast_dims)

    if out is not None:
        assert result_dtype is not None
        if dtype is not None and result_dtype != out.dtype:
            raise RuntimeError(
                "Expected the dtype of reduction result and out to match"
            )
        out = _maybe_resize_out(out, result.shape)
        return _safe_copy_out(copy_from=result, copy_to=out)  # type: ignore[arg-type]

    if result.dtype != result_dtype and result_dtype is not None:
        result = prims.convert_element_type(result, result_dtype)

    return result


def _make_copy_from_view(fn):
    """
    Given a view function (e.g. torch.diagonal) generates its copy variant (e.g. torch.diagonal_copy)
    """
    aten_fn = getattr(aten, fn.__name__)
    annotations = getattr(fn, "__annotations__", {})
    fn = out_wrapper()(aten_fn)

    @wraps(fn)
    def _fn(*args, out=None, **kwargs):
        result = fn(*args, out=out, **kwargs)
        if out is not None:
            return result

        return pytree.tree_map(
            lambda x: x.clone(memory_format=torch.contiguous_format),
            result,
        )

    copy_name = f"{fn.__name__}_copy"
    _fn.__name__ = copy_name
    _fn.__annotations__.update(annotations)
    register_decomposition(getattr(aten, copy_name))(_fn)
    return _fn


@register_decomposition(aten.all)
@out_wrapper()
def all(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
) -> TensorLikeType:
    result = torch.logical_not(torch.any(torch.logical_not(a), dim, keepdim=keepdim))

    if a.dtype == torch.uint8:
        result = result.to(dtype=torch.uint8)

    return result


@register_decomposition(aten.any)
@out_wrapper()
def any(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
) -> TensorLikeType:
    a_ = _maybe_convert_to_dtype(a, torch.bool)
    if isinstance(dim, (list, tuple)) and len(dim) == 0:
        result = a_.clone()
    else:
        result = a_.sum(dim=dim, keepdim=keepdim).ne(False)

    # Preserves uint8 -- probably a legacy mask thing
    if a.dtype is torch.uint8:
        return prims.convert_element_type(result, torch.uint8)

    return result


@register_decomposition([aten.sum.dim_IntList, aten.sum.IntList_out])
def sum(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    if dtype is None:
        if out is not None:
            dtype = out.dtype
        elif utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
            dtype = torch.int64
        else:
            dtype = a.dtype
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None
    return _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=out,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


def sum_to_size(
    a: Tensor,
    *shape,
) -> Tensor:
    shape = utils.extract_shape_from_varargs(shape, validate=False)
    torch._check(
        utils.is_expandable_to(shape, a.shape),
        lambda: f'sum_to_size: size "{shape}" is not expandable to size "{a.shape}"',
    )
    # In ATen scalar tensors are sent through sum and the result is returned as
    # type promoted
    if utils.is_same_shape(shape, a.shape) and len(shape) > 0:
        return prims.view_of(a)
    leading_dims = a.ndim - len(shape)
    reduce_dims = tuple(range(leading_dims)) + tuple(
        i
        for i in range(leading_dims, len(shape))
        if shape[i - leading_dims] == 1 and a.shape[i] != 1
    )
    return torch.sum(a, dim=reduce_dims, keepdim=True, dtype=None)


@register_decomposition(aten.prod)
def prod(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    dtype=None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    if dtype is None:
        if out is not None:
            dtype = out.dtype
        elif utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
            dtype = torch.int64
        else:
            dtype = a.dtype
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None
    return _reduction(
        a,
        prims.prod,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=out,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


@register_decomposition(aten.amin)
def amin(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None

    return _reduction(
        a,
        prims.amin,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        out=out,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


@register_decomposition(aten.amax)
def amax(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None

    return _reduction(
        a,
        prims.amax,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        out=out,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


def _dim_var_dispatch(dim=None, unbiased=None):
    # There's the following overload of torch.var:
    # var(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
    # We need to explicitly convert bool dims to unbiased arg
    if unbiased is None and isinstance(dim, bool):
        unbiased = dim
        dim = None
    return dim, unbiased


@register_decomposition(aten.var)
@out_wrapper()
def var(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[NumberType] = None,
) -> TensorLikeType:
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    correction = utils.set_correction(unbiased, correction)
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None

    result = _reduction(
        a,
        partial(prims.var, correction=correction),
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        out=None,
        has_identity=True,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    )
    return result


@register_decomposition(aten.std)
@out_wrapper()
def std(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[NumberType] = None,
) -> TensorLikeType:
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    correction = utils.set_correction(unbiased, correction)

    opmath_dtype, dtype = utils.reduction_dtypes(
        a, REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    )
    a = _maybe_convert_to_dtype(a, opmath_dtype)
    a_var = torch.var(a, dim, correction=correction, keepdim=keepdim)
    a_std = torch.sqrt(a_var)
    assert dtype is not None
    return _maybe_convert_to_dtype(a_std, dtype)


@register_decomposition(aten.mean)
def mean(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    dtype=None,
    out=None,
) -> TensorLikeType:
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None
    orig_dtype = dtype
    if dtype is None:
        dtype = a.dtype
    # can't use out wrapper because of this argument
    torch._check(
        out is None or out.dtype == dtype,
        lambda: f"Expected out tensor to have dtype {dtype}, but got {out.dtype} instead",
    )
    result = _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=None,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE,
    )
    torch._check(
        utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype),
        lambda: (
            f"mean(): could not infer output dtype. "
            f"{'Input' if orig_dtype is None else 'Optional'} dtype must be either "
            f"a floating point or complex dtype. Got: {dtype}"
        ),
    )
    if isinstance(dim, Dim):
        dim = (dim,)  # type: ignore[assignment]
    dims = utils.reduction_dims(a.shape, dim)  # type: ignore[arg-type]
    nelem = 1 if a.ndim == 0 else reduce(operator.mul, (a.shape[i] for i in dims), 1)
    result = true_divide(result, nelem)
    result_dtype = a.dtype if dtype is None else dtype
    result = _maybe_convert_to_dtype(result, result_dtype)  # type: ignore[method-assign]
    if out is not None:
        assert isinstance(out, TensorLike)
        out = _maybe_resize_out(out, result.shape)
        return _safe_copy_out(copy_from=result, copy_to=out)  # type: ignore[arg-type]
    return result


@register_decomposition(aten.std_mean)
@out_wrapper("out0", "out1")
def std_mean(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    *,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    correction: Optional[NumberType] = None,
):
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    correction = utils.set_correction(unbiased, correction)
    opmath_dtype, dtype = utils.reduction_dtypes(
        a, REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    )
    original_dtype = a.dtype
    a = _maybe_convert_to_dtype(a, opmath_dtype)
    a_var, a_mean = torch.var_mean(a, dim, correction=correction, keepdim=keepdim)
    a_std = torch.sqrt(a_var)
    assert dtype is not None
    return (
        _maybe_convert_to_dtype(a_std, dtype),
        _maybe_convert_to_dtype(a_mean, original_dtype),
    )


@register_decomposition(aten.var_mean)
@out_wrapper("out0", "out1")
def var_mean(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[NumberType] = None,
):
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    v = var(a, dim, unbiased, keepdim, correction=correction)
    m = mean(a, dim, keepdim)
    return v, m


@register_decomposition(aten.addr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "vec1", "vec2"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def addr(
    self: TensorLikeType,
    vec1: TensorLikeType,
    vec2: TensorLikeType,
    *,
    beta: NumberType = 1,
    alpha: NumberType = 1,
) -> TensorLikeType:
    torch._check(
        vec1.ndim == 1,
        lambda: f"addr: Expected 1-D argument vec1, but got {vec1.ndim}-D",
    )
    torch._check(
        vec2.ndim == 1,
        lambda: f"addr: Expected 1-D argument vec2, but got {vec2.ndim}-D",
    )
    for arg, arg_name in ((alpha, "alpha"), (beta, "beta")):
        if isinstance(arg, bool):
            torch._check(
                utils.is_boolean_dtype(self.dtype)
                and utils.is_boolean_dtype(vec1.dtype)
                and utils.is_boolean_dtype(vec2.dtype),
                lambda: f"Boolean {arg_name} only supported for Boolean results.",
            )
    self = self.expand(vec1.shape[0], vec2.shape[0])
    if utils.is_boolean_dtype(self.dtype):
        # Integers are accepted for booleans
        torch._check(
            is_weakly_lesser_type(type(beta), int),
            lambda: f"expected bool/int beta but got {type(beta)}",
        )
        torch._check(
            is_weakly_lesser_type(type(alpha), int),
            lambda: f"expected bool/int alpha but got {type(beta)}",
        )
        if not beta:
            return torch.outer(vec1, vec2) if alpha else torch.full_like(self, False)
        else:
            return torch.logical_or(
                self,
                torch.outer(vec1, vec2) if alpha else torch.full_like(self, False),
            )
    else:
        torch._check(
            is_weakly_lesser_type(type(beta), dtype_to_type(self.dtype)),
            lambda: f"cannot safely convert {type(beta)} to {self.dtype}",
        )
        torch._check(
            is_weakly_lesser_type(type(alpha), dtype_to_type(self.dtype)),
            lambda: f"cannot safely convert {type(alpha)} to {self.dtype}",
        )
        if beta == 0:
            # This means NaNs from self are dropped if beta is zero
            return alpha * torch.outer(vec1, vec2)
        else:
            return beta * self + alpha * torch.outer(vec1, vec2)


# CompositeImplicitAutograd - don't register decomp
def atleast_1d(
    arg: Union[TensorLikeType, Sequence[TensorLikeType]], *args: TensorLikeType
) -> Union[TensorLikeType, Tuple[TensorLikeType, ...]]:
    """Reference implementation of :func:`torch.atleast_1d`."""
    if not args and isinstance(arg, collections.abc.Sequence):
        args_ = arg
    else:
        assert not isinstance(arg, collections.abc.Sequence)
        args_ = (arg,) + args
    res = tuple(a if a.ndim >= 1 else unsqueeze(a, 0) for a in args_)
    return res if len(res) > 1 else res[0]


# Helper function with assert to avoid MyPy error
# of incompatible type passed to unsqueeze
def _unsqueeze_atleast(
    at_least_fn: Callable, dim: int, arg: TensorLikeType
) -> TensorLikeType:
    arg_ = at_least_fn(arg)
    assert isinstance(arg_, TensorLike)
    return unsqueeze(arg_, dim)


# CompositeImplicitAutograd - don't register decomp
def atleast_2d(
    arg: Union[TensorLikeType, Sequence[TensorLikeType]], *args: TensorLikeType
) -> Union[TensorLikeType, Tuple[TensorLikeType, ...]]:
    """Reference implementation of :func:`torch.atleast_2d`."""
    if not args and isinstance(arg, collections.abc.Sequence):
        args_ = arg
    else:
        assert not isinstance(arg, collections.abc.Sequence)
        args_ = (arg,) + args
    unsqueeze_atleast_1d = partial(_unsqueeze_atleast, atleast_1d, 0)
    res = tuple(a if a.ndim >= 2 else unsqueeze_atleast_1d(a) for a in args_)
    return res if len(res) > 1 else res[0]


# CompositeImplicitAutograd - don't register decomp
def atleast_3d(
    arg: Union[TensorLikeType, Sequence[TensorLikeType]], *args: TensorLikeType
) -> Union[TensorLikeType, Tuple[TensorLikeType, ...]]:
    """Reference implementation of :func:`torch.atleast_3d`."""
    if not args and isinstance(arg, collections.abc.Sequence):
        args_ = arg
    else:
        assert not isinstance(arg, collections.abc.Sequence)
        args_ = (arg,) + args
    unsqueeze_atleast_2d = partial(_unsqueeze_atleast, atleast_2d, -1)
    res = tuple(a if a.ndim >= 3 else unsqueeze_atleast_2d(a) for a in args_)
    return res if len(res) > 1 else res[0]


def as_strided(
    a: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: Optional[int] = None,
) -> TensorLikeType:
    storage_offset_int = (
        storage_offset if storage_offset is not None else a.storage_offset()
    )
    return prims.as_strided(a, size, stride, storage_offset_int)


@register_decomposition(aten.as_strided_scatter)
@out_wrapper()
def as_strided_scatter(
    input: TensorLikeType,
    src: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: Optional[int] = None,
) -> TensorLikeType:
    storage_offset_int = 0 if storage_offset is None else storage_offset
    return prims.as_strided_scatter(input, src, size, stride, storage_offset_int)


def broadcast_shapes(*shapes) -> ShapeType:
    return torch.Size(_broadcast_shapes(*shapes))


@aten.broadcast_tensors.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.broadcast_tensors.default.py_impl(DispatchKey.Meta)
def broadcast_tensors(*tensors) -> List[TensorLikeType]:
    if len(tensors) == 1 and not isinstance(tensors[0], Tensor):
        tensors = tensors[0]
    return list(_maybe_broadcast(*tensors, preserve_cpu_scalar_tensors=False))


# CompositeImplicitAutograd - don't register decomp
def broadcast_to(a: TensorLikeType, size: ShapeType) -> TensorLikeType:
    start = len(size) - len(a.shape)
    dims = tuple(range(start, len(a.shape) + start))
    return prims.broadcast_in_dim(a, size, dims)


@register_decomposition(aten.cat)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("tensors",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def cat(tensors: TensorSequenceType, dim: int = 0) -> TensorLikeType:
    def cat_compute_output_memory_format(inputs):
        format = None
        for t in inputs:
            f = utils.suggest_memory_format(t)
            if f == torch.contiguous_format:
                return f
            if format is not None and format != f:
                return torch.contiguous_format
            format = f
        assert format is not None
        return format

    if len(tensors) == 0:
        msg = "cat expects at least one tensor, but received zero!"
        raise ValueError(msg)

    for tensor in tensors:
        assert isinstance(tensor, TensorLike)

    utils.check_same_device(*tensors, allow_cpu_scalar_tensors=False)

    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # This is a bit tricky.  Naively, you would expect to just pick one
    # arbitrary tensor and check that all tensors match this tensor.  However,
    # there is legacy behavior which says that if you have a 1-D empty tensor
    # (0,), this is permissible.  So you can't assume that all the tensors
    # have same dimensionality, and you can't assume that the first tensor is
    # the correct stencil.
    #
    # We'll implement this in a few passes.  First, we will try to infer the
    # ndim of the cat output.  If this ndim != 1, then we know that all ndim =
    # 1 inputs must be empty, or are errors.  If this ndim == 1, then life
    # is easy (the legacy special case coincides with regular handling).
    #
    # NB: The regular implementation of cat just filters out empty inputs,
    # but we do it slightly different here for better handling for unbacked
    # SymInts

    example = None
    for i, t in enumerate(tensors):
        if example is None:
            if t.ndim != 1:
                example = t
        else:
            if t.ndim != 1:
                torch._check(
                    t.ndim == example.ndim,
                    lambda: "Number of dimensions of tensors must match.  "
                    f"Expected {example.ndim}-D tensors, but got {t.ndim}-D for "
                    f"tensor number {i} in the list",
                )

    if example is None:
        # example is None if everything is 1-D.  If so, just arbitrarily pick
        # the first one
        example = tensors[0]

    shape = example.shape
    filtered = []
    for tensor_idx, tensor in enumerate(tensors):
        if len(shape) != len(tensor.shape):
            assert tensor.ndim == 1  # we've already checked this above
            # Don't suggest the legacy behavior in the error message
            torch._check(
                tensor.shape[0] == 0,
                lambda: f"Number of dimensions of tensors must match.  "
                f"Expected {example.ndim}-D tensors, but got 1-D for "
                f"tensor number {tensor_idx} in the list",
            )
        else:
            # Remove inputs that are 1-D, zero size
            if tensor.ndim == 1 and guard_size_oblivious(tensor.shape[0] == 0):
                continue
            # Don't bother checking size match, prims.cat will handle it
            filtered.append(tensor)

    memory_format = cat_compute_output_memory_format(tensors)

    if len(filtered) == 0:
        t = tensors[0]

        # TODO: fix this to work with meta tensors
        try:
            # BUG? This looks like it wants to call builtins.any() but is
            # actually calling .any() (in this file). Changing to builtins.any()
            # causes tests to fail:
            # PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=4 python test/test_ops.py -k \
            #   TestFakeTensorCUDA.test_fake_crossref_backward_amp_cat_cuda_float32
            requires_grad = bool(any(x.requires_grad for x in tensors))  # type: ignore[arg-type]
        except Exception:
            requires_grad = False  # type: ignore[assignment]

        return empty(
            (0,),
            dtype=t.dtype,
            device=t.device,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )

    dim = utils.canonicalize_dim(filtered[0].ndim, dim)
    utils.validate_idx(filtered[0].ndim, dim)

    return prims.cat(filtered, dim).clone(memory_format=memory_format)


# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
def column_stack(tensors: TensorSequenceType) -> TensorLikeType:
    aligned_tensors = tuple(
        x if x.ndim > 1 else x.reshape((x.numel(), 1)) for x in tensors
    )
    return cat(aligned_tensors, 1)


def conj(input: TensorLikeType) -> TensorLikeType:
    if not utils.is_complex_dtype(input.dtype):
        return input
    if input.is_sparse:
        return torch.conj_physical(input)
    return prims.conj(input)


# This replicates at::constant_pad_nd, defined in ATen/native/PadNd.cpp
@register_decomposition(aten.constant_pad_nd)
@out_wrapper()
def constant_pad_nd(
    input: TensorLikeType, pad: List[int], value: NumberType = 0
) -> TensorLikeType:
    torch._check(
        len(pad) % 2 == 0,
        lambda: f"Length of pad must be even but instead it equals {len(pad)}",
    )

    input_sizes = input.shape
    l_inp = len(input_sizes)

    l_pad = len(pad) // 2
    l_diff = l_inp - l_pad

    torch._check(
        l_inp >= l_pad,
        lambda: "Length of pad should be no more than twice the number of "
        f"dimensions of the input. Pad length is {len(pad)} while the input has "
        f"{l_inp} dimensions.",
    )

    c_input = input
    for i in range(l_diff, l_inp):
        pad_idx = 2 * (l_inp - i - 1)
        if pad[pad_idx] < 0:
            c_input = c_input.narrow(i, -pad[pad_idx], c_input.shape[i] + pad[pad_idx])

        if pad[pad_idx + 1] < 0:
            c_input = c_input.narrow(i, 0, c_input.shape[i] + pad[pad_idx + 1])

    # if none of the pads are positive we can just return the result
    if builtins.all(p <= 0 for p in pad):
        return c_input.clone()

    new_shape = list(input_sizes[:l_diff])

    for i in range(l_pad):
        pad_idx = len(pad) - ((i + 1) * 2)
        new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1]
        torch._check(
            new_dim > 0,
            lambda: f"The input size {input_sizes[l_diff + i]}, plus negative padding "
            f"{pad[pad_idx]} and {pad[pad_idx + 1]} resulted in a negative output size, "
            f"which is invalid. Check dimension {l_diff + i} of your input.",
        )
        new_shape.append(new_dim)

    memory_format = utils.suggest_memory_format(input)
    output = torch.empty(
        new_shape,
        dtype=input.dtype,
        device=input.device,
        requires_grad=input.requires_grad,
        memory_format=memory_format,
    )

    if value == 0 and input.dtype == torch.bool:
        value = False
    # torch.fill isn't typed to allow complex values
    output = torch.fill(output, value)  # type: ignore[arg-type]

    c_output = output
    for i in range(l_diff, l_inp):
        pad_idx = 2 * (l_inp - i - 1)
        if pad[pad_idx] > 0:
            c_output = c_output.narrow(
                i, pad[pad_idx], c_output.shape[i] - pad[pad_idx]
            )
        if pad[pad_idx + 1] > 0:
            c_output = c_output.narrow(i, 0, c_output.shape[i] - pad[pad_idx + 1])

    prims.copy_to(c_output, c_input)
    return output


def contiguous(
    a: Tensor, *, memory_format: torch.memory_format = torch.contiguous_format
) -> Tensor:
    torch._check(
        memory_format != torch.preserve_format,
        lambda: "preserve memory format is unsupported by the contiguous operator",
    )

    if utils.is_contiguous_for_memory_format(a, memory_format=memory_format):
        return a

    return torch.clone(a, memory_format=memory_format)


@out_wrapper()
def dstack(tensors: TensorSequenceType) -> TensorLikeType:
    torch._check(len(tensors) > 0, lambda: "dstack expects a non-empty TensorList")
    aligned_tensors = atleast_3d(*tensors)
    return cat(aligned_tensors, 2)


@register_decomposition(aten.expand)
def expand(a: Tensor, *shape) -> Tensor:
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # NOTE: cannot use utils.extract_shape_from_varargs here
    # because that also validates the shape, but the shape
    # given to expand may be "invalid"
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = tuple(shape[0])

    torch._check(
        len(shape) >= len(a.shape),
        lambda: "expand: the requested shape has too few dimensions!",
    )

    offset = len(shape) - len(a.shape)
    shape_ = list(shape)
    for idx, x in enumerate(a.shape):
        offset_idx = idx + offset
        requested_length = shape[offset_idx]
        torch._check(
            guard_size_oblivious(requested_length == x)
            or guard_size_oblivious(x == 1)
            or requested_length == -1,
            lambda: f"expand: attempting to expand a dimension of length {x}!",
        )

        shape_[offset_idx] = requested_length if requested_length != -1 else x

    # At this point shape must be valid
    utils.validate_shape(shape_)

    return prims.broadcast_in_dim(
        a, shape_, tuple(range(offset, len(a.shape) + offset))
    )


# CompositeImplicitAutograd - don't register decomp
def expand_as(a: Tensor, b: Tensor) -> Tensor:
    return a.expand(b.shape)


def chunk(a: TensorLikeType, chunks: int, dim: int = 0) -> Tuple[TensorLikeType, ...]:
    if chunks <= 0:
        msg = f"Expected at least one chunk, but got {chunks}!"
        raise ValueError(msg)

    dim = utils.canonicalize_dim(a.ndim, dim)
    length = a.shape[dim]
    chunk_size = math.ceil(length / chunks)
    full_chunks = math.floor(length / chunk_size)
    tail_chunk_size = length % chunk_size

    result = []
    for i in range(full_chunks):
        result.append(narrow(a, dim, i * chunk_size, chunk_size))

    if tail_chunk_size != 0:
        result.append(narrow(a, dim, full_chunks * chunk_size, tail_chunk_size))

    return tuple(result)


# Note: flatten, unlike other shape operators, returns the input tensor on a no-op (unless
# a 0D tensor is flattened, in which case it's returned in 1D)
# CompositeImplicitAutograd - don't register decomp
def flatten(a: TensorLikeType, start_dim: int = 0, end_dim: int = -1) -> TensorLikeType:
    start_dim = utils.canonicalize_dim(a.ndim, start_dim)
    end_dim = utils.canonicalize_dim(a.ndim, end_dim)

    # Short-circuits on no-op
    if start_dim == end_dim and a.ndim != 0:
        return a

    # Tries to take a view
    # TODO: we could look at directing collapse_view to skip its meta function here (unsafe_collapse_view)
    new_shape, new_strides = prims._collapse_view_helper(a, start_dim, end_dim)
    if new_shape is not None:
        return prims.collapse_view(a, start_dim, end_dim)

    # Makes a copy if it can't make a view
    return prims.collapse(a, start_dim, end_dim)


@register_decomposition(aten.flip)
@out_wrapper()
def flip(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType:
    if not isinstance(dims, tuple) and not isinstance(dims, list):
        raise ValueError("dims has to be a sequence of ints")
    dims = utils.canonicalize_dims(a.ndim, dims)  # type: ignore[assignment]
    utils.validate_no_repeating_dims(dims)
    return prims.rev(a, dims)


# CompositeImplicitAutograd - don't register decomp
def fliplr(a: TensorLikeType) -> TensorLikeType:
    if a.ndim < 2:
        raise RuntimeError("Input must be >= 2-d.")

    return flip(a, (1,))


# CompositeImplicitAutograd - don't register decomp
def flipud(a: TensorLikeType) -> TensorLikeType:
    if a.ndim < 1:
        raise RuntimeError("Input must be >= 1-d.")

    return flip(a, (0,))


# CompositeImplicitAutograd - don't register decomp
def narrow(
    a: TensorLikeType, dim: int, start: Union[int, TensorLikeType], length: int
) -> TensorLikeType:
    # Supports Tensor overload that was added for XLA:
    # https://github.com/pytorch/pytorch/issues/31558
    if isinstance(start, TensorLike):
        torch._check(
            start.dim() == 0 and utils.is_integer_dtype(start.dtype),
            lambda: "start must be an 0-dim integral Tensor.",
        )
        start = start.item()  # type: ignore[assignment]
    torch._check(a.dim() > 0, lambda: "narrow() cannot be applied to a 0-dim tensor.")
    torch._check(length >= 0, lambda: "narrow(): length must be non-negative.")
    dim = utils.canonicalize_dim(a.ndim, dim)
    dim_length = a.size(dim)
    torch._check_with(
        IndexError,
        -dim_length <= start and start <= dim_length,  # type: ignore[arg-type]
        lambda: f"start out of range (expected to be in range of [{-dim_length}, {dim_length}], but got {start})",
    )
    if start < 0:
        start = start + dim_length
    torch._check(
        start <= dim_length - length,  # type: ignore[arg-type]
        lambda: f"start ({start}) + length ({length}) exceeds dimension size ({dim_length}).",
    )
    return prims.slice_in_dim(a, start, start + length, axis=dim)


def _normalize(
    a: Tensor, norm_dims: DimsType, eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes mean and 1/std of a tensor along norm_dims.

    Used as a helper function for normalization layers.

    Args:
        a (Tensor): input tensor
        norm_dims (DimsType): dimensions to normalize over
        eps (float): epsilon for numerical stability

    Returns:
        out (Tensor): normalized tensor.
        mean (Tensor): mean of the tensor along norm_dims.
        rstd (Tensor): 1/std of the tensor along norm_dims.
    """
    norm_dims = utils.canonicalize_dims(a.ndim, norm_dims)
    computation_dtype = utils.get_computation_dtype(a.dtype)
    a_acc = _maybe_convert_to_dtype(a, computation_dtype)
    assert isinstance(a_acc, TensorLike)  # to avoid mypy error for var_mean
    biased_var, mean = torch.var_mean(
        a_acc, dim=norm_dims, unbiased=False, keepdim=True
    )
    rstd = torch.rsqrt(biased_var + eps)
    out = (a - mean) * rstd
    return out, mean, rstd


# add all specified dimensions
def _unsqueeze_multiple(x: TensorLikeType, dimensions: List[int]) -> TensorLikeType:
    for dim in sorted(dimensions):
        x = torch.unsqueeze(x, dim)
    return x


@register_decomposition(aten.native_group_norm.default)
def native_group_norm(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    batch_size: int,
    num_channels: int,
    flattened_inner_size: int,
    num_groups: int,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    torch._check(
        input.ndim >= 2,
        lambda: f"Expected at least 2 dimensions for input tensor but received {input.ndim}",
    )
    torch._check(
        num_channels % num_groups == 0,
        lambda: "Expected number of channels in input to be divisible by num_groups, "
        + f"but got input of shape {input.shape} and num_groups = {num_groups}",
    )

    # num_channels / num_groups and flattened inner dimension are the reduction axes
    reduction_dims = [2, 3]
    input_reshaped = torch.reshape(
        input,
        [batch_size, num_groups, num_channels // num_groups, flattened_inner_size],
    )
    out, mean, rstd = _normalize(input_reshaped, reduction_dims, eps)
    out = out.view(input.shape)

    broadcast_dims = [0] + list(range(2, input.ndim))
    unsqueeze_bias = None
    if bias is not None:
        unsqueeze_bias = _unsqueeze_multiple(bias, broadcast_dims)
    unsqueeze_weight = None
    if weight is not None:
        unsqueeze_weight = _unsqueeze_multiple(weight, broadcast_dims)

    if unsqueeze_weight is not None:
        out = out * unsqueeze_weight
    if unsqueeze_bias is not None:
        out = out + unsqueeze_bias

    out = _maybe_convert_to_dtype(out, input.dtype)  # type: ignore[assignment]
    mean = _maybe_convert_to_dtype(mean, input.dtype)  # type: ignore[assignment]
    rstd = _maybe_convert_to_dtype(rstd, input.dtype)  # type: ignore[assignment]

    # remove broadcast dimensions from mean and rstd
    mean = torch.squeeze(mean, reduction_dims)
    rstd = torch.squeeze(rstd, reduction_dims)
    return (out, mean, rstd)


@register_decomposition(aten.native_layer_norm)
@out_wrapper("out0", "out1", "out2")
def native_layer_norm(
    input: Tensor,
    normalized_shape: ShapeType,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    normalized_ndim = len(normalized_shape)
    torch._check(
        normalized_ndim >= 1,
        lambda: "Expected normalized_shape to be at least 1-dimensional, i.e., "
        + "containing at least one element, but got normalized_shape = "
        + str(normalized_shape),
    )
    # torch.Size([1, 2, 3]) == [1, 2, 3] evaluates to False
    # while torch.Size([1, 2, 3]) == (1, 2, 3) is True
    # therefore we use tuple(normalized_shape)
    torch._check(
        weight is None or weight.shape == tuple(normalized_shape),
        lambda: "Expected weight to be of same shape as normalized_shape, but got "
        + "weight of shape "
        + str(weight.shape)  # type: ignore[union-attr]
        + " and normalized_shape = "
        + str(normalized_shape),
    )
    torch._check(
        bias is None or bias.shape == tuple(normalized_shape),
        lambda: "Expected bias to be of same shape as normalized_shape, but got "
        + "bias of shape "
        + str(bias.shape)  # type: ignore[union-attr]
        + " and normalized_shape = "
        + str(normalized_shape),
    )
    torch._check(
        input.ndim >= normalized_ndim
        and input.shape[(input.ndim - normalized_ndim) :] == tuple(normalized_shape),
        lambda: "Given normalized_shape="
        + str(normalized_shape)
        + ", expected input with shape "
        + str(normalized_shape)
        + ", but got input of size "
        + str(input.shape),
    )

    input = input.contiguous()
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    axis = input.ndim - normalized_ndim
    reduction_dims = list(range(axis, input.ndim))
    out, mean, rstd = _normalize(input, reduction_dims, eps)

    if weight is None and bias is not None:
        out = out + bias
    elif weight is not None and bias is None:
        out = out * weight
    elif weight is not None and bias is not None:
        out = out * weight + bias

    out = _maybe_convert_to_dtype(out, input.dtype)  # type: ignore[assignment]
    if input.device.type == "cpu":
        mean = _maybe_convert_to_dtype(mean, input.dtype)  # type: ignore[assignment]
        rstd = _maybe_convert_to_dtype(rstd, input.dtype)  # type: ignore[assignment]
    return (out, mean, rstd)


# TODO: Adding this as a meta function causes functorch tests to fail when compiled with debug mode.
# test/test_eager_transforms.py::TestFunctionalizeCPU::test_functionalize_fx_transpose_simple_cpu
@register_decomposition(aten.permute)
def permute(a: TensorLikeType, *dims) -> TensorLikeType:
    _permutation = utils.canonicalize_dims(
        a.ndim, utils.extract_dims_from_varargs(dims)
    )
    return prims.transpose(a, _permutation)


@register_decomposition(aten.renorm)
@out_wrapper()
def renorm(
    input: TensorLikeType, p: RealNumberType, dim: int, maxnorm: RealNumberType
) -> TensorLikeType:
    torch._check(not isinstance(p, complex), lambda: "renorm: p must be real-valued")
    torch._check(p > 0, lambda: "renorm: non-positive norm not supported")
    torch._check(
        not isinstance(maxnorm, complex), lambda: "renorm: maxnorm must be real-valued"
    )
    torch._check(
        maxnorm >= 0, lambda: f"renorm: expected maxnorm to be >= 0 but got {maxnorm}"
    )
    ndim = input.ndim
    torch._check(
        ndim > 1,
        lambda: f"renorm: input needs at least 2 dimensions, got {ndim} dimensions",
    )

    dim = utils.canonicalize_dim(ndim, dim)
    reduce_dims = list(range(ndim))
    del reduce_dims[dim]

    # For half and bfloat16, calculate norm in float precision then cast
    # normalization factor to half
    acc_type = utils.get_computation_dtype(input.dtype)
    if acc_type != input.dtype:
        norm = torch.linalg.vector_norm(
            input, p, reduce_dims, keepdim=True, dtype=acc_type
        )
    else:
        norm = torch.linalg.vector_norm(input, p, reduce_dims, keepdim=True)

    eps = 1e-7
    norm_factor = torch.where(norm > maxnorm, maxnorm / (norm + eps), 1.0)
    if acc_type != input.dtype:
        norm_factor = prims.convert_element_type(norm_factor, input.dtype)
    return (input * norm_factor).contiguous()


# CompositeImplicitAutograd - don't register decomp
@aten.stft.center.py_impl(DispatchKey.CompositeImplicitAutograd)
def stft(
    input: Tensor,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[Tensor] = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: Optional[bool] = None,
    return_complex: Optional[bool] = None,
) -> Tensor:
    torch._check(
        window is None or window.device == input.device,
        lambda: (
            f"stft input and window must be on the same device but got self on {input.device}"
            + f" and window on {window.device}"  # type: ignore[union-attr]
        ),
    )

    hop_length_ = hop_length if hop_length is not None else n_fft // 4
    win_length_ = win_length if win_length is not None else n_fft

    if return_complex is None:
        return_complex_ = input.is_complex() or (
            window is not None and utils.is_complex_dtype(window.dtype)
        )
        torch._check(
            return_complex_,
            (
                "stft requires the return_complex parameter be given for real inputs, "
                + "and will further require that return_complex=True in a future PyTorch release."
            ),
        )
    else:
        return_complex_ = return_complex

    torch._check(
        utils.is_float_dtype(input.dtype) or utils.is_complex_dtype(input.dtype),
        lambda: "stft expected a tensor of floating point or complex values",
    )
    torch._check(1 <= input.ndim <= 2, lambda: "stft expected a 1D or 2D tensor")

    original_ndim = input.ndim
    if original_ndim == 1:
        input = input.unsqueeze(0)

    if center:
        extra_dims = 3 - input.ndim
        pad_amount = n_fft // 2
        extended_shape = [*itertools.repeat(1, extra_dims), *input.shape]
        input = aten.pad(input.view(extended_shape), [pad_amount, pad_amount], pad_mode)
        input = input.view(input.size()[extra_dims:])

    batch = input.size(0)
    length = input.size(1)
    torch._check(
        0 < n_fft <= length,
        lambda: f"stft expected 0 < n_fft <= {length}, but got n_fft={n_fft}",
    )
    torch._check(
        hop_length_ > 0,
        lambda: f"stft expected hop_length > 0 but got hop_length={hop_length_}",
    )
    torch._check(
        0 < win_length_ <= n_fft,
        lambda: f"stft expected 0 < win_length <= n_fft but got win_length={win_length_}",
    )
    torch._check(
        window is None or window.shape == (win_length_,),
        lambda: (
            f"expected a 1D window tensor of size equal to win_length={win_length_}, "
            + f"but got window with size {window.shape}"  # type: ignore[union-attr]
        ),
    )

    if win_length_ < n_fft:
        if window is None:
            window = torch.ones(win_length_, dtype=input.dtype, device=input.device)
        left = (n_fft - win_length_) // 2
        window = aten.constant_pad_nd(window, [left, n_fft - win_length_ - left])

    input = input.unfold(dimension=-1, size=n_fft, step=hop_length_)
    if window is not None:
        input = input * window

    complex_fft = utils.is_complex_dtype(input.dtype)
    onesided = onesided if onesided is not None else not complex_fft
    norm = "ortho" if normalized else None
    if onesided:
        torch._check(
            not complex_fft,
            lambda: "Cannot have onesided output if window or input is complex",
        )
        out = torch.fft.rfft(input, dim=-1, norm=norm)
    else:
        out = torch.fft.fft(input, dim=-1, norm=norm)

    out.transpose_(1, 2)

    if original_ndim == 1:
        out = out.squeeze_(0)

    return out if return_complex_ else torch.view_as_real(out)


# CompositeImplicitAutograd - don't register decomp
@aten.istft.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def istft(
    input: Tensor,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[Tensor] = None,
    center: bool = True,
    normalized: bool = False,
    onesided: Optional[bool] = None,
    length: Optional[int] = None,
    return_complex=False,
) -> Tensor:
    torch._check(
        window is None or window.device == input.device,
        lambda: (
            f"istft input and window must be on the same device but got self on {input.device}"
            + f" and window on {window.device}"  # type: ignore[union-attr]
        ),
    )

    hop_length_ = hop_length if hop_length is not None else n_fft // 4
    win_length_ = win_length if win_length is not None else n_fft

    torch._check(
        utils.is_complex_dtype(input.dtype),
        lambda: (
            "istft input and window must be on the same device but got self on "
            + f"{input.device} and window on {window.device}"  # type: ignore[union-attr]
        ),
    )
    n_frames = input.size(-1)
    fft_size = input.size(-2)

    expected_output_signal_len = n_fft + hop_length_ * (n_frames - 1)
    torch._check(input.numel() > 0, lambda: "istft input tensor cannot be empty")
    torch._check(
        2 <= input.ndim <= 3,
        lambda: f"istft expected a tensor with 2 or 3 dimensions, but got {input.ndim}",
    )
    onesided_ = onesided if onesided is not None else fft_size != n_fft

    if onesided_:
        torch._check(
            n_fft // 2 + 1 == fft_size,
            lambda: (
                "istft expected the frequency dimension (3rd to the last) of the input tensor "
                + "to match n_fft / 2 + 1 when onesided=True, but got {fft_size}"
            ),
        )
    else:
        torch._check(
            n_fft == fft_size,
            lambda: (
                "istft expected the frequency dimension (3rd to the last) of the input tensor "
                + "to match n_fft when onesided=False, but got {fft_size}",
            ),
        )

    torch._check(
        0 < hop_length_ <= win_length_,
        lambda: "istft expected 0 < hop_length <= win_length",
    )
    torch._check(
        0 < win_length_ <= n_fft, lambda: "istft expected 0 < win_length <= n_fft"
    )
    torch._check(
        window is None or window.shape == (win_length_,),
        lambda: "Invalid window shape. window has to be 1D and length of `win_length`",
    )

    if window is None:
        real_dtype = utils.corresponding_real_dtype(input.dtype)
        window_ = torch.ones(win_length_, dtype=real_dtype, device=input.device)
    else:
        window_ = window

    if win_length_ != n_fft:
        left = (n_fft - win_length_) // 2
        window_ = aten.constant_pad_nd(window_, (left, n_fft - win_length_ - left), 0)

    original_ndim = input.ndim
    if input.ndim == 2:
        input = input.unsqueeze(0)

    input = input.transpose(1, 2)
    norm = "ortho" if normalized else None
    if return_complex:
        torch._check(
            not onesided_,
            lambda: "cannot have onesided output if window or input is complex",
        )
        input = torch.fft.ifft(input, dim=-1, norm=norm)
    else:
        torch._check(
            window is None or not utils.is_complex_dtype(window.dtype),
            lambda: "Complex windows are incompatible with return_complex=False",
        )
        if not onesided_:
            input = input.narrow(dim=-1, start=0, length=n_fft // 2 + 1)
        input = torch.fft.irfft(input, dim=-1, norm=norm)

    assert input.size(2) == n_fft

    y_tmp = input * window_.view([1, 1, n_fft])
    y = aten.unfold_backward(
        y_tmp,
        input_sizes=(y_tmp.size(0), expected_output_signal_len),
        dim=1,
        size=n_fft,
        step=hop_length_,
    )
    window_envelop = aten.unfold_backward(
        window_.pow(2).expand((1, n_frames, n_fft)),
        input_sizes=(y_tmp.size(0), expected_output_signal_len),
        dim=1,
        size=n_fft,
        step=hop_length_,
    )

    assert expected_output_signal_len == y.size(1)
    assert expected_output_signal_len == window_envelop.size(1)

    start = n_fft // 2 if center else 0
    if length is not None:
        end = start + length
    elif center:
        end = expected_output_signal_len - n_fft // 2
    else:
        end = expected_output_signal_len

    length = max(0, end - start)
    y = y.narrow(dim=1, start=start, length=length)
    window_envelop = window_envelop.narrow(dim=1, start=start, length=length)

    window_envelop_lowest = window_envelop.abs().min().lt(1e-11)
    torch._check(
        not window_envelop_lowest.item(),
        lambda: "window overlap add min less than 1e-11",
    )

    y = y / window_envelop
    if original_ndim == 2:
        y = y.squeeze(0)

    if end > expected_output_signal_len:
        warnings.warn(
            "The length of signal is shorter than the length parameter. Result is being "
            + "padded with zeros in the tail. Please check your center and hop_length settings"
        )
        y = aten.constant_pad_nd(y, (0, end - expected_output_signal_len), 0)
    return y


# Get the new shape and stride after applying unfold to an input tensor
def _get_unfold_shape_stride(
    a_shape: ShapeType, a_stride: StrideType, dimension: int, size: int, step: int
):
    a_ndim = len(a_shape)
    dim = utils.canonicalize_dim(a_ndim, dimension, wrap_scalar=True)
    max_size = 1 if a_ndim == 0 else a_shape[dim]
    last_stride = 1 if a_ndim == 0 else a_stride[dim]

    torch._check(
        size <= max_size,
        lambda: f"Maximum size for tensor at dimension {dim} is {max_size} but size is {size}",
    )

    torch._check(
        step > 0,
        lambda: f"Step is {step} but must be > 0",
    )

    shape = list(a_shape)
    strides = list(a_stride)
    shape.append(size)
    strides.append(last_stride)
    if dim < a_ndim:
        shape[dim] = (shape[dim] - size) // step + 1
        strides[dim] *= step
    return shape, strides


@register_decomposition(aten.repeat)
@out_wrapper()
def repeat(a: Tensor, *repeat_shape) -> Tensor:
    repeat_shape = utils.extract_shape_from_varargs(repeat_shape, validate=False)
    torch._check(
        len(repeat_shape) >= len(a.shape),
        lambda: "repeat: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor",
    )

    if len(repeat_shape) == 0:
        return torch.clone(a)

    num_new_dimensions = len(repeat_shape) - a.ndim
    padded_shape = [1] * num_new_dimensions
    for dim_size in a.shape:
        padded_shape.append(dim_size)

    target_shape = tuple(
        padded_size * repeat_size
        for padded_size, repeat_size in zip(padded_shape, repeat_shape)
    )

    # return an empty tensor if one of the repeat_shape dimensions is zero
    if 0 in repeat_shape:
        return torch.empty(
            target_shape,
            dtype=a.dtype,
            device=a.device,
            requires_grad=a.requires_grad,
            memory_format=utils.suggest_memory_format(a),
        )

    urtensor_shape = target_shape
    urtensor_stride = utils.make_contiguous_strides_for(target_shape)
    for dim, dim_size in enumerate(padded_shape):
        # repeat each dimension by using unfold_copy operation
        urtensor_shape, urtensor_stride = _get_unfold_shape_stride(
            urtensor_shape, urtensor_stride, dim, dim_size, max(dim_size, 1)
        )

    # derive permute order by sorting urtensor strides
    enumerated_stride = list(enumerate(urtensor_stride))
    enumerated_stride.sort(key=operator.itemgetter(1), reverse=True)
    permute_order, sorted_stride = zip(*enumerated_stride)

    # add new and expand dimensions according to urtensor
    repeat_xtensor = a.expand(urtensor_shape)

    # clone tensor to concretize expanded dimensions
    cloned_result = torch.clone(repeat_xtensor)

    # transpose axis so strides are in sorted order
    permuted_result = cloned_result.permute(permute_order)

    # reshape to get contiguous tensor with correct target shape
    return permuted_result.reshape(target_shape)


def _reshape_view_helper(a: TensorLikeType, *shape, allow_copy: bool) -> TensorLikeType:
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious, sym_eq

    # Creates a valid shape
    shape = utils.extract_shape_from_varargs(shape, validate=False)
    # Reshape may be given a shape with a -1 length
    # This indicates that the dimension's length should be inferred
    shape = utils.infer_size(shape, a.numel())

    # Special-cases tensors with no elements
    if guard_size_oblivious(a.numel() == 0):
        return as_strided(a, shape, utils.make_contiguous_strides_for(shape))

    # Special-cases reshaping zero dim tensors
    if a.ndim == 0:
        _a = a
        for length in shape:
            assert length == 1
            _a = unsqueeze(_a, -1)
        if _a is a:
            return prims.view_of(a)
        else:
            return _a

    # Special-cases reshaping to zero dim tensors
    if len(shape) == 0:
        _a = a
        for length in a.shape:
            assert length == 1
            _a = squeeze(_a, -1)
        if _a is a:
            return prims.view_of(a)
        else:
            return _a

    if a.is_contiguous():
        # Special-cases for nd_to_1d
        if len(shape) == 1 and a.ndim > 1:
            return torch.as_strided(a, [a.numel()], [1])
        # Special-cases for 1d_to_2d
        if len(shape) == 2 and a.ndim == 1:
            dim0 = shape[0]
            dim1 = shape[1]
            return torch.as_strided(a, [dim0, dim1], [dim1, 1])

    # Handles general case: a 1+D tensor reshaped into a distinct 1+D shape

    # NOTE [Reshape Algorithm]
    # This algorithm works by attempting to greedily construct the desired dimensions in
    # the output shape, left to right. It does this by, conceptually, accumulating
    # dimensions of the original tensor, also left to right, until the dimension
    # can be constructed using prims.split_dim.
    # The algorithm also has special handling for tail squeezes/unsqueezes, like
    # if a reshape from (5, 5) to (5, 5, 1) or vice versa.
    #
    # This algorithm does not flatten the original tensor and then split dims as appropriate
    # because that would create copies more often than this algorithm. flatten is the only
    # operation below which can create a view or a copy, and while it prefers creating
    # views it may sometimes create a copy if the tensor's strides do not permit a view.
    # As a result, this algorithm tries to minimize flattening.
    #
    # Note that a better version of this algorithm may exist. Regions which could be
    # flattened without creating a copy can be identified in advance, and that might
    # allow fewer flatten calls or faster short-circuiting to make a copy.
    idx = 0
    a_ = a
    for length in shape:
        # Handles tail unsqueezes
        if idx >= a_.ndim:
            assert length == 1
            last_dim = a_.ndim - 1
            # NOTE: using split_dim instead of unsqueeze may seem silly here,
            # but it's necessary to get the strides correct
            a_ = prims.split_dim(a_, last_dim, a_.shape[last_dim])
            idx = idx + 1
            continue

        # Skips dimensions that are already the correct length
        if guard_size_oblivious(length == a_.shape[idx]):
            idx = idx + 1
            continue

        # Gathers enough original dimensions such that this new dimension can be created
        # Note that this accumulation will terminate because we've verified a and the shape
        # specify the same number of elements above
        accum = a_.shape[idx]
        end = idx
        while guard_size_oblivious(accum % length != 0):
            end = end + 1
            accum = accum * a_.shape[end]
        if end != idx:
            # NOTE: in this case multiple dimensions must be flatten to create the desired dimension
            # This flattening is why reshape sometimes creates a copy -- because flattening
            # may return a view of a copy

            # Checks if collapse can be a view and short-circuits to copying reshape if it can't
            new_shape, new_strides = prims._collapse_view_helper(a_, idx, end)
            if new_shape is None:
                if allow_copy:
                    return prims.reshape(a, shape)

                msg = f"Cannot view a tensor with shape {a.shape} and strides {a.stride()} as a tensor with shape {shape}!"
                raise ValueError(msg)

            a_ = flatten(a_, idx, end)

        # Splits the (possibly flattened) dimension to create the desired dim length
        if guard_size_oblivious(accum != length):
            a_ = prims.split_dim(a_, idx, length)

        idx = idx + 1

    # Squeezes tail
    while idx < a_.ndim:
        torch._check(
            a_.shape[idx] == 1,
            lambda: f"a.size({idx}) expected to be 1 but got {a_.shape[idx]}",
        )
        a_ = squeeze(a_, idx)

    if a_ is a:
        return prims.view_of(a)
    else:
        return a_


# CompositeImplicitAutograd - don't register decomp
# NOTE: shape is a vararg because Tensor.reshape can be called with as
# Tensor.reshape(a, b, c) or Tensor.reshape((a, b, c)) Function call
# torch.reshape doesn't support unpacked shapes
def reshape(a: TensorLikeType, *shape: ShapeType) -> TensorLikeType:
    return _reshape_view_helper(a, *shape, allow_copy=True)


# CompositeImplicitAutograd - don't register decomp
def reshape_as(self: TensorLikeType, other: TensorLikeType) -> TensorLikeType:
    return self.reshape(other.size())


@register_decomposition(aten.roll)
@out_wrapper()
def roll(a: TensorLikeType, shifts: DimsType, dims: DimsType = ()) -> TensorLikeType:
    """Reference implementation of :func:`torch.roll`."""
    dims = utils.canonicalize_dims(a.ndim, dims)
    # ATen specifies int[1] type for shifts and dims which expands integers to tuples of length 1
    if not isinstance(shifts, Iterable):
        shifts = (shifts,)
    if not isinstance(dims, Iterable):
        dims = (dims,)

    # Avoid modulo by zero
    if a.numel() == 0:
        # Keeping this as ref for now as FakeTensor runs into some issues with complex tensors
        return a.clone()

    if a.dim() == 0 and len(dims) > 0:
        raise IndexError(
            f"Dimension specified as {dims[0]} but tensor has no dimensions"
        )

    len_shifts = len(shifts)
    len_dims = len(dims)
    if len_shifts != 1 or len_dims != 1:
        if len_shifts == 0:
            raise RuntimeError("`shifts` required")
        # Takes care of the case when dims is not specified (default)
        # By default, the tensor is flattened before shifting, after which the original shape is restored
        if len_dims == 0 and len_shifts == 1:
            return torch.roll(torch.flatten(a), shifts, 0).view(a.shape)
        if len_shifts != len_dims:
            raise RuntimeError(
                f"shifts and dimensions must align. shifts: {len_shifts}, dims: {len_dims}"
            )
        assert len_dims > 1
        tail_shifts = shifts[1:]
        tail_dims = dims[1:]
        first_dim_rolled = torch.roll(a, (shifts[0],), dims[0])
        return torch.roll(first_dim_rolled, tail_shifts, tail_dims)

    # This path is taken when only one dimension is rolled
    # For example to get `first_dim_rolled` above
    dim = dims[0]
    size = a.shape[dim]
    start = (size - shifts[0]) % size
    idx = torch.arange(size, device=a.device)
    return a.index_select(dim, torch.fmod(start + idx, size))


@register_decomposition(aten.rot90)
@out_wrapper()
def rot90(
    a: TensorLikeType, k: int = 1, dims: DimsSequenceType = (0, 1)
) -> TensorLikeType:
    """Reference implementation of :func:`torch.rot90`."""
    if len(dims) != 2:
        raise RuntimeError(
            f"expected total rotation dims == 2, but got dims = {len(dims)}"
        )
    if a.ndim < 2:
        raise RuntimeError(f"expected total dims >= 2, but got total dims = {a.ndim}")

    # Do this after the initial checks to be compatible with the behavior in
    # core.
    dims = utils.canonicalize_dims(a.ndim, dims)

    if dims[0] == dims[1]:
        raise RuntimeError(
            f"expected rotation dims to be different, but got dim0 = {dims[0]} and dim1 = {dims[1]}"
        )
    k = k % 4  # Rotation direction is from the second towards the first axis for k < 0
    if k == 1:
        return torch.transpose(torch.flip(a, (dims[1],)), dims[0], dims[1])
    elif k == 2:
        return torch.flip(a, dims)
    elif k == 3:
        return torch.transpose(torch.flip(a, (dims[0],)), dims[0], dims[1])
    else:
        return a.clone(memory_format=torch.contiguous_format)


def _check_stack_inputs(tensors: TensorSequenceType) -> None:
    entry_shape = tensors[0].shape
    for i in range(1, len(tensors)):
        assert tensors[i].shape == entry_shape, (
            f"stack expects each tensor to be equal size, but got {entry_shape} at entry 0 "
            f"and {tensors[i].shape} at entry {i}"
        )


@register_decomposition(aten.stack)
@out_wrapper()
def stack(tensors: TensorSequenceType, dim: int = 0) -> TensorLikeType:
    assert len(tensors) > 0, "stack expects a non-empty TensorList"
    wrapped_dim = utils.canonicalize_dim(tensors[0].ndim + 1, dim)
    # Refs need sparse support to check other condition
    if wrapped_dim < tensors[0].ndim:  # and not tensors[0].is_sparse:
        _check_stack_inputs(tensors)
        result_sizes = list(tensors[0].shape)
        result_sizes.insert(wrapped_dim, len(tensors))
        out = torch.cat(tensors, wrapped_dim)
        return out.view(result_sizes)

    # If dim == tensors[0].ndim, view cannot efficiently handle it
    return torch.cat([t.unsqueeze(wrapped_dim) for t in tensors], dim)


# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
def softmax(
    a: TensorLikeType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    result_dtype = dtype or a.dtype
    computation_dtype = utils.get_computation_dtype(result_dtype)
    a_ = _maybe_convert_to_dtype(a, computation_dtype)
    if a.numel() == 0:
        a_exp = exp(a_)
    else:
        a_max = amax(a_, dim, keepdim=True)
        a_exp = exp(a_ - a_max)
    return _maybe_convert_to_dtype(
        true_divide(a_exp, sum(a_exp, dim, keepdim=True)), result_dtype
    )  # type: ignore[return-value]


# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
def hstack(tensors: TensorSequenceType) -> TensorLikeType:
    torch._check(len(tensors) > 0, lambda: "hstack expects a non-empty TensorList")
    aligned_tensors = atleast_1d(*tensors)
    if aligned_tensors[0].ndim == 1:
        return cat(aligned_tensors, 0)
    return cat(aligned_tensors, 1)


# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
def vstack(tensors: TensorSequenceType) -> TensorLikeType:
    torch._check(len(tensors) > 0, lambda: "vstack expects a non-empty TensorList")
    aligned_tensors = atleast_2d(*tensors)
    return cat(aligned_tensors, 0)


# CompositeImplicitAutograd - don't register decomp
def unflatten(a: TensorLikeType, dim: int, sizes: ShapeType) -> TensorLikeType:
    dim = utils.canonicalize_dim(a.ndim, dim)
    torch._check(len(sizes) != 0, lambda: "unflatten: sizes must be non-empty")
    return a.view(tuple(a.shape[:dim]) + tuple(sizes) + tuple(a.shape[dim + 1 :]))


@register_decomposition(aten.unbind)
def unbind(t: TensorLikeType, dim: int = 0) -> TensorSequenceType:
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    dim = utils.canonicalize_dim(t.ndim, dim)
    torch._check_index(
        len(t.shape) > 0,
        lambda: "Dimension specified as 0 but tensor has no dimensions",
    )
    if guard_size_oblivious(t.shape[dim] == 0):
        return ()
    else:
        return tuple(
            torch.squeeze(s, dim) for s in torch.tensor_split(t, t.shape[dim], dim)
        )


@out_wrapper()
def index_copy(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike):
    return x.clone(memory_format=torch.contiguous_format).index_copy_(
        dim, index, tensor
    )


def index_copy_(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike):
    dim = utils.canonicalize_dims(x.ndim, dim)
    torch._check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    # Treat scalars as elements of \R^1
    y = x.unsqueeze(0) if x.ndim == 0 else x
    idx = (slice(None),) * dim + (index,)
    y[idx] = tensor
    return x


@register_decomposition(aten.index_fill)
@out_wrapper()
def index_fill(
    x: TensorLike, dim: int, index: TensorLike, value: Union[NumberType, TensorLike]
):
    return _index_fill(x, dim, index, value, inplace=False)


@register_decomposition(aten.index_fill_)
def index_fill_(
    x: TensorLike, dim: int, index: TensorLike, value: Union[NumberType, TensorLike]
):
    return _index_fill(x, dim, index, value, inplace=True)


def _index_fill(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    value: Union[NumberType, TensorLike],
    *,
    inplace: bool,
):
    torch._check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    if isinstance(value, TensorLike):
        torch._check(
            value.ndim == 0,
            lambda: "Only supports 0-dimensional value tensor. "  # type: ignore[union-attr]
            f"Got a tensor with {value.ndim} dimensions.",
        )  # type: ignore[arg-type]
    else:
        value = torch.scalar_tensor(
            value, dtype=x.dtype, layout=x.layout, device=x.device  # type: ignore[arg-type]
        )

    # index_copy has some unnecessary preconditions when x is a scalar. We do this to work through them
    zero_dim = x.ndim == 0
    y = x.unsqueeze(0) if zero_dim else x
    # index_copy does not broadcast on value so we have to do it manually
    shape = list(y.shape)
    shape[dim] = index.numel()
    value = value.expand(shape)
    index_copy = Tensor.index_copy_ if inplace else torch.index_copy
    out = index_copy(y, dim, index, value)  # type: ignore[operator]
    if inplace:
        return x
    else:
        if zero_dim:
            # The clone is necessary so that it returns a fresh tensor rather than a view
            out = out.squeeze(0).clone()
        # index_fill preserves the strides. index_copy always returns contiguous tensors
        if out.stride() != x.stride():
            new_out = torch.empty_like(x)
            new_out.copy_(out)
            out = new_out
        return out


@out_wrapper()
def index_add(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    alpha: NumberType = 1,
):
    # index_add always returns a new contiguous tensor
    return x.clone(memory_format=torch.contiguous_format).index_add_(
        dim, index, tensor, alpha=alpha  # type: ignore[arg-type]
    )


@register_decomposition(aten.index_select)
@out_wrapper()
def index_select(x: TensorLike, dim: int, index: TensorLike):
    dim = utils.canonicalize_dims(x.ndim, dim)
    torch._check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    if index.ndim == 0:
        index = index.unsqueeze(0)
    if x.ndim == 0:
        # Treat scalars as elements of \R^1
        # We cannot use x[idx] here as it accesses item() (??), hence this awkward construction
        return torch.empty_like(x).index_copy(0, index, x.expand_as(index))

    idx = (slice(None),) * dim + (index,)
    return x[idx]


@register_decomposition(aten.squeeze.dims)
def squeeze(a: TensorLikeType, dim: Optional[DimsType] = None) -> TensorLikeType:
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    if dim is None:
        dims = tuple(idx for idx, size in enumerate(a.shape) if size == 1)
        return prims.squeeze(a, dims) if dims else prims.view_of(a)

    ndim = a.ndim
    dim = utils.canonicalize_dims(ndim, dim)
    dims = (dim,) if isinstance(dim, Dim) else dim
    # Short-circuits if the tensor has no dimensions
    if ndim == 0:
        assert len(dims) == 0 or dims == (0,)
        return prims.view_of(a)

    # Note: squeeze does not modify tensors when the given dim is not a dimension of length 1
    dims = tuple(d for d in dims if guard_size_oblivious(a.shape[d] == 1))
    if len(dims) == 0:
        return prims.view_of(a)
    if len(dims) == 1:
        return prims.squeeze(a, dims)
    dims_list = list(dims)
    dims_list = sorted(dims_list, reverse=True)
    for i in dims_list:
        a = squeeze(a, i)
    return a


# Note: does not work with TensorMetas because of data-dependent control-flow
# CompositeImplicitAutograd - don't register decomp
def tensor_split(
    a: TensorLikeType,
    indices_or_sections: Union[Tensor, DimsType],
    dim: int = 0,
) -> Tuple[TensorLikeType, ...]:
    _dim = utils.canonicalize_dim(a.ndim, dim)
    if a.ndim == 0:
        msg = "tensor_split: received a rank zero tensor, but expected a tensor of rank one or greater!"
        raise ValueError(msg)

    # If indices_or_sections is a tensor, it must be a CPU Long tensor
    if isinstance(indices_or_sections, TensorLike):
        if not indices_or_sections.device.type == "cpu":
            msg = (
                f"tensor_split: if indices_or_sections is a tensor it must be on the CPU, "
                f"but received one on {indices_or_sections.device}"
            )
            raise ValueError(msg)
        if indices_or_sections.dtype != torch.long:
            msg = "tensor_split: if indices_or_sections is a tensor it must have long dtype, "
            f" but received one with dtype {indices_or_sections.dtype}"
            raise ValueError(msg)

    # Case 0 -- indices_or_sections is an integer or a scalar tensor n and a is split along dim into n parts of equal-ish length
    if isinstance(indices_or_sections, IntLike) or (
        isinstance(indices_or_sections, TensorLike) and indices_or_sections.ndim == 0
    ):
        sections: int = (
            indices_or_sections  # type: ignore[assignment]
            if isinstance(indices_or_sections, Number)
            else indices_or_sections.item()
        )

        if sections <= 0:
            msg = f"tensor_split: number of sections must be greater than 0, but was {sections}"
            raise ValueError(msg)

        splits = []
        dim_size = a.shape[_dim]
        min_split_size = math.floor(dim_size / sections)
        num_splits_one_extra = dim_size % sections
        start_idx = 0
        for split_idx in range(sections):
            split_size = (
                min_split_size + 1
                if (split_idx < num_splits_one_extra)
                else min_split_size
            )
            s = prims.slice_in_dim(a, start_idx, start_idx + split_size, axis=_dim)
            splits.append(s)
            start_idx = start_idx + split_size

        return tuple(splits)
    # Case 1 -- indices_or_sections is a sequence of integers or a 1D tensor describing the splits
    else:
        indices = indices_or_sections
        if isinstance(indices_or_sections, TensorLike):
            if indices_or_sections.ndim != 1:
                msg = "tensor_split: non-scalar indices_or_sections tensors must have only one dimension, "
                f"but received a tensor with {indices_or_sections.ndim} dimensions"
                raise ValueError(msg)

            indices = indices_or_sections.tolist()

        splits = []
        start_idx = 0
        for x in indices:
            splits.append(prims.slice_in_dim(a, start_idx, x, axis=_dim))
            start_idx = x
        splits.append(prims.slice_in_dim(a, start_idx, a.shape[_dim], axis=_dim))
        return tuple(splits)


# CompositeImplicitAutograd - don't register decomp
def hsplit(
    a: TensorLikeType, indices_or_sections: DimsType
) -> Tuple[TensorLikeType, ...]:
    torch._check(
        a.ndim >= 1,
        lambda: (
            "torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with "
            + str(a.ndim)
            + " dimensions!"
        ),
    )
    dim = 0 if a.ndim == 1 else 1
    if isinstance(indices_or_sections, IntLike):
        split_size = indices_or_sections
        torch._check(
            (split_size != 0 and a.shape[dim] % split_size == 0),
            lambda: (
                "torch.hsplit attempted to split along dimension "
                + str(dim)
                + ", but the size of the dimension "
                + str(a.shape[dim])
                + " is not divisible by the split_size "
                + str(split_size)
                + "!"
            ),
        )
        return tensor_split(a, split_size, dim)

    torch._check_type(
        isinstance(indices_or_sections, (list, tuple)),
        lambda: (
            "hsplit(): received an invalid combination of arguments. "
            "Expected indices_or_sections to be of type int, list of ints or tuple of ints "
            f"but got type {type(indices_or_sections)}"
        ),
    )

    split_sizes = indices_or_sections
    return tensor_split(a, split_sizes, dim)


# CompositeImplicitAutograd - don't register decomp
def vsplit(
    a: TensorLikeType, indices_or_sections: DimsType
) -> Tuple[TensorLikeType, ...]:
    torch._check(
        a.ndim >= 2,
        lambda: (
            "torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with "
            + str(a.ndim)
            + " dimensions!"
        ),
    )
    if isinstance(indices_or_sections, IntLike):
        split_size = indices_or_sections
        torch._check(
            (split_size != 0 and a.shape[0] % split_size == 0),
            lambda: (
                f"torch.vsplit attempted to split along dimension 0"
                f", but the size of the dimension "
                f"{a.shape[0]}"
                f" is not divisible by the split_size "
                f"{split_size}"
                f"!"
            ),
        )
        return tensor_split(a, split_size, 0)

    torch._check_type(
        isinstance(indices_or_sections, (list, tuple)),
        lambda: (
            "vsplit(): received an invalid combination of arguments. "
            "Expected indices_or_sections to be of type int, list of ints or tuple of ints "
            f"but got type {type(indices_or_sections)}"
        ),
    )

    split_sizes = indices_or_sections
    return tensor_split(a, split_sizes, 0)


@register_decomposition(aten.diag.out)
@out_wrapper()
def diag(
    self: TensorLikeType,
    offset: int = 0,
) -> TensorLikeType:
    ndim = self.dim()
    torch._check(
        ndim in (1, 2), lambda: f"diag(): Supports 1D or 2D tensors. Got {ndim}D"
    )
    if ndim == 1:
        return torch.diag_embed(self, offset)
    else:
        return torch.diagonal_copy(self, offset)


@register_decomposition(aten.diagonal_scatter)
@out_wrapper()
def diagonal_scatter(
    input: TensorLikeType,
    src: TensorLikeType,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
) -> TensorLikeType:
    out = utils.clone_preserve_strides(input)
    diag = out.diagonal(offset, dim1, dim2)
    torch._check(
        diag.shape == src.shape,
        lambda: "expected src to have a size equal to the diagonal of the input."
        f"Got {src.shape} for a diagonal of shape {diag.shape}",
    )
    copy_to(diag, src)
    return out


@register_decomposition(aten.diagonal)
def diagonal(
    self: TensorLikeType,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
) -> TensorLikeType:
    """
    Reference implementation of torch.diagonal
    """
    num_dims = self.dim()
    dim1 = utils.canonicalize_dim(idx=dim1, rank=num_dims)
    dim2 = utils.canonicalize_dim(idx=dim2, rank=num_dims)

    torch._check(
        dim1 != dim2, lambda: f"diagonal dimensions cannot be identical {dim1}, {dim2}"
    )

    storage_offset = self.storage_offset()

    if offset >= 0:
        diag_size = max(min(self.size()[dim1], self.size()[dim2] - offset), 0)
    else:
        diag_size = max(min(self.size()[dim1] + offset, self.size()[dim2]), 0)

    if diag_size > 0:
        if offset >= 0:
            storage_offset += offset * self.stride()[dim2]
        else:
            storage_offset -= offset * self.stride()[dim1]

    sizes = [s for i, s in enumerate(self.size()) if i not in (dim1, dim2)]
    sizes.append(diag_size)

    strides = [s for i, s in enumerate(self.stride()) if i not in (dim1, dim2)]
    strides.append(self.stride()[dim1] + self.stride()[dim2])

    result = self.as_strided(size=sizes, stride=strides, storage_offset=storage_offset)

    return result


@register_decomposition(aten.diag_embed)
@out_wrapper()
def diag_embed(
    t: TensorLikeType,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> TensorLikeType:
    """
    Reference implementation of torch.diag_embed
    """
    # convert from negative dims
    rank = t.ndim + 1
    dim1 = utils.canonicalize_dim(rank=rank, idx=dim1)
    dim2 = utils.canonicalize_dim(rank=rank, idx=dim2)

    # as per the docs, exchanging dims is equivalent to changing the sign of
    # offset
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
        offset = -offset

    torch._check(
        dim1 != dim2, lambda: f"diagonal dimensions cannot be identical {dim1}, {dim2}"
    )

    # as per the docs, the size of last dim is placed at dim1 and dim2
    last_dim = t.size(-1)

    if offset != 0:
        # add padding to match the new size
        t_shape = list(t.shape)
        t_shape[-1] = builtins.abs(offset)
        z = torch.zeros(t_shape, dtype=t.dtype, device=t.device, requires_grad=False)
        pair = (z, t) if offset > 0 else (t, z)
        t = torch.cat(pair, dim=-1)
        # make sure the diagonal always has the same size
        last_dim += builtins.abs(offset)

    # preserve original data, but place 1 at dim1 and move last dim to dim2
    t = t.unsqueeze(dim1).movedim(-1, dim2)

    # generate ranges shifting indices based on offset
    a_range = torch.arange(last_dim, device=t.device, dtype=torch.int64)
    b_range = torch.arange(
        offset, last_dim + offset, device=t.device, dtype=torch.int64
    )

    # broadcast
    cond = a_range == b_range.unsqueeze(-1)
    cond_shape = [last_dim if i in (dim1, dim2) else 1 for i in range(len(t.shape))]
    cond = cond.reshape(cond_shape)

    # aten.diag_embed always returns a new contiguous tensor
    # contiguous() is needed to correctly model the output stride
    return utils.mask_tensor(cond, t).contiguous()


@register_decomposition(aten.block_diag)
@out_wrapper()
def _block_diag_iterable(tensors: List[TensorLikeType]) -> TensorLikeType:
    """
    Reference implementation of torch.block_diag
    """
    tensors_2d = [
        tensor.view(1, -1) if tensor.dim() <= 1 else tensor for tensor in tensors
    ]

    ncols = builtins.sum(tensor.shape[1] for tensor in tensors_2d)
    device = tensors_2d[0].device

    result = []

    col_start = 0
    for i, tensor in enumerate(tensors_2d):
        torch._check(
            tensor.dim() == 2,
            lambda: "Input tensors must have 2 or fewer dimensions. "
            f"Input {i} has {tensor.dim()} dimensions",
        )
        torch._check(
            tensor.device == device,
            lambda: "Input tensors must all be on the same device. "
            f"Input 0 is on device {device} and input {i} is on device {tensor.device}.",
        )
        row, col = tensor.shape
        left = torch.zeros((row, col_start), device=device, dtype=tensor.dtype)
        right = torch.zeros(
            (row, ncols - col_start - col), device=device, dtype=tensor.dtype
        )
        result += [torch.cat((left, tensor, right), dim=1)]
        col_start += col

    return torch.cat(result, dim=0)


def block_diag(*tensors: List[TensorLikeType]) -> TensorLikeType:
    """
    This is used as an input to PythonRefInfo. `torch.block_diag`
    expects arguments splatted, but `aten.block_diag` expects only
    one argument that is a list of Tensors.
    """
    return _block_diag_iterable(tensors)  # type: ignore[arg-type]


# CompositeImplicitAutograd - don't register decomp
def dsplit(a: TensorLikeType, sections: DimsType) -> TensorSequenceType:
    if a.ndim < 3:
        raise RuntimeError(
            f"torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with {a.ndim} dimensions!"
        )
    if isinstance(sections, IntLike) and (sections == 0 or a.shape[2] % sections != 0):
        raise RuntimeError(
            "torch.dsplit attempted to split along dimension 2, "
            + f"but the size of the dimension {a.shape[2]} is not divisible by the split_size {sections}!"
        )
    return tensor_split(a, sections, 2)


@register_decomposition(aten.t.default)
def t(a: TensorLikeType):
    # TODO: Add sparse support
    # if a.is_sparse:
    #     sparse_dim = a.sparse_dim()
    #     dense_dim = a.dense_dim()
    #     if not (sparse_dim <= 2 and dense_dim == 0):
    #         raise RuntimeError(
    #             f"t() expects a tensor with <= 2 sparse and 0 dense dimensions, but got {sparse_dim} sparse and"
    #             f"{dense_dim} dense dimensions"
    #         )
    if a.ndim > 2:
        raise RuntimeError(
            f"t() expects a tensor with <= 2 dimensions, but self is {a.ndim}D"
        )
    return torch.transpose(a, 0, 0 if a.ndim < 2 else 1)


# CompositeImplicitAutograd - don't register decomp
def T(a: TensorLikeType) -> TensorLikeType:
    # n != 2 && n != 0 is deprecated in regular PyTorch.
    torch._check(
        a.ndim in (0, 2),
        lambda: (
            "The use of `x.T` on tensors of dimension other than 0 or 2 "
            "to reverse their shape is not supported."
        ),
    )
    return a.t()


@register_decomposition(aten.alias)
def alias(a: TensorLikeType) -> TensorLikeType:
    return prims.view_of(a)


@register_decomposition(aten.transpose)
def transpose(a: TensorLikeType, dim0: int, dim1: int) -> TensorLikeType:
    _dim0, _dim1 = utils.canonicalize_dims(a.ndim, (dim0, dim1))  # type: ignore[misc]

    if a.ndim <= 1 or dim0 == dim1:
        return aten.alias.default(a)

    _permutation = list(range(0, a.ndim))
    _permutation[_dim0] = _dim1
    _permutation[_dim1] = _dim0
    return torch.permute(a, _permutation)


# Aliases for transpose
swap_axes = transpose


@register_decomposition(aten.unfold)
def unfold(
    self: TensorLikeType, dimension: int, size: int, step: int
) -> TensorLikeType:
    shape, strides = _get_unfold_shape_stride(
        self.shape, self.stride(), dimension, size, step
    )
    return self.as_strided(shape, strides)


@register_decomposition(aten.unfold_copy)
@out_wrapper()
def unfold_copy(self: TensorLikeType, dimension: int, size: int, step: int):
    return self.unfold(dimension, size, step).clone(
        memory_format=torch.contiguous_format
    )


def _cumsumprod_common(
    func,
    init,
    a: TensorLikeType,
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # We implement all the kwargs of a reduction. ATen just handles dtype
    # nb. This decomposition may not be as efficient as a backend-specific implementation
    ndim = a.ndim
    dim = utils.canonicalize_dim(ndim, dim)
    if ndim == 0:
        return func(a.unsqueeze(0), dim=0, dtype=dtype, out=out)
    a = a.unsqueeze(dim + 1)
    rg = torch.arange(a.shape[dim], device=a.device)
    mask = rg.unsqueeze(1) <= rg
    for _ in range(ndim - dim - 1):
        mask = mask.unsqueeze(-1)
    masked_a = torch.where(mask, a, init)
    return func(masked_a, dim=dim, dtype=dtype, out=out)


@register_decomposition(aten.cumsum)
def cumsum(
    a: TensorLikeType,
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    return _cumsumprod_common(func=sum, init=0, a=a, dim=dim, dtype=dtype, out=out)


@register_decomposition(aten.cumprod)
def cumprod(
    a: TensorLikeType,
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    return _cumsumprod_common(func=prod, init=1, a=a, dim=dim, dtype=dtype, out=out)


# Note: although squeeze is documented as having the out= kwarg it doesn't
@register_decomposition(aten.unsqueeze)
def unsqueeze(a: TensorLikeType, dim: int) -> TensorLikeType:
    # Note that unsqueeze canonicalizes with rank + 1 because it allows
    # a new innermost dimension to be specified
    ndim = a.ndim + 1
    dim = utils.canonicalize_dim(ndim, dim)
    return prims.expand_dims(a, (dim,), ndim=ndim)


# NOTE: shape is a vararg because Tensor.reshape can be called with as
# Tensor.view(a, b, c) or Tensor.view((a, b, c)) Function call torch.view
# doesn't support unpacked shapes
# TODO: Turn this into a decomposition (currently fails on reshape meta tests)
@register_decomposition(aten.view.default)
def view(a: TensorLikeType, *shape: ShapeType) -> TensorLikeType:
    return _reshape_view_helper(a, *shape, allow_copy=False)


# CompositeImplicitAutograd - don't register decomp
def view_as(self: TensorLikeType, other: TensorLikeType) -> TensorLikeType:
    return self.view(other.size())


# CompositeImplicitAutograd - don't register decomp
def ravel(a: TensorLikeType) -> TensorLikeType:
    return reshape(a, (-1,))


# CompositeImplicitAutograd - don't register decomp
# missing ref impl. for aten.gather
@out_wrapper()
def take_along_dim(
    a: torch.Tensor, indices: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    torch._check(
        a.ndim == indices.ndim,
        lambda: (
            "torch.take_along_dim(): input and indices should have the same "
            f"number of dimensions, but got {a.ndim} dimensions for input, and "
            f"{indices.ndim} dimensions for indices"
        ),
    )

    torch._check(
        utils.is_integer_dtype(indices.dtype),
        lambda: (
            "torch.take_along_dim(): dtype of indices should be int but got "
            f"{indices.dtype} instead"
        ),
    )

    if dim is None:
        return torch.gather(a.view(-1), 0, indices.view(-1))
    else:
        self_sizes = list(a.shape)
        self_sizes[dim] = indices.size(dim)
        broadcast_shape = utils.infer_size_shapes(self_sizes, indices.size())
        indices_broadcast = broadcast_to(indices, broadcast_shape)

        indices_sizes = list(indices.shape)
        indices_sizes[dim] = a.size(dim)
        broadcast_shape = utils.infer_size_shapes(indices_sizes, a.size())
        self_broadcast = broadcast_to(a, broadcast_shape)

        return torch.gather(self_broadcast, dim, indices_broadcast)


@out_wrapper()
def empty(
    *shape,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> TensorLikeType:
    torch._check(
        memory_format != torch.preserve_format,
        lambda: "torch.empty: the Preserve memory format is not supported",
    )

    shape = utils.extract_shape_from_varargs(shape)

    if memory_format == torch.contiguous_format:
        strides = utils.make_contiguous_strides_for(shape)
    elif memory_format == torch.channels_last_3d:
        strides = utils.make_channels_last_3d_strides_for(shape)
    else:  # memory_format == torch.channels_last
        torch._check(
            memory_format == torch.channels_last,
            lambda: f"torch.empty: received an unknown memory format {memory_format}!",
        )
        strides = utils.make_channels_last_2d_strides_for(shape)

    return torch.empty_strided(
        shape,
        strides,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@out_wrapper()
def empty_permuted(
    shape,
    physical_layout,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> TensorLikeType:
    return prims.empty_permuted(
        shape,
        physical_layout,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


@register_decomposition(aten.new_empty)
@out_wrapper()
def new_empty(
    a: TensorLikeType,
    size: ShapeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
) -> TensorLikeType:
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    return torch.empty(
        size,
        dtype=dtype,
        device=device,
        pin_memory=pin_memory,
        layout=layout,
    )


@register_decomposition(aten.new_empty_strided)
@out_wrapper()
def new_empty_strided(
    a: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.Tensor.new_empty_strided
    """

    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    return torch.empty_strided(
        size,
        stride,
        dtype=dtype,
        device=device,
        pin_memory=pin_memory,
        layout=layout,
    )


@register_decomposition(aten.zeros.default)
@out_wrapper()
def zeros(
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    size = utils.extract_shape_from_varargs(size)

    if dtype is None:
        dtype = torch.get_default_dtype()

    return torch.full(
        size,
        False if dtype == torch.bool else 0,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_decomposition(aten.new_zeros)
@out_wrapper()
def new_zeros(
    a: TensorLikeType,
    size: ShapeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    return torch.full(
        size,
        False if (dtype or a.dtype) == torch.bool else 0,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_decomposition(aten.ones.default)
@out_wrapper()
def ones(
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    size = utils.extract_shape_from_varargs(size)

    if dtype is None:
        dtype = torch.get_default_dtype()

    return torch.full(
        size,
        True if dtype == torch.bool else 1,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_decomposition(aten.new_ones)
@out_wrapper()
def new_ones(
    a: TensorLikeType,
    size: ShapeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    return torch.full(
        size,
        True if (dtype or a.dtype) == torch.bool else 1,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_decomposition(aten.new_full)
@out_wrapper()
def new_full(
    a: TensorLikeType,
    size: ShapeType,
    fill_value: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
) -> TensorLikeType:
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    return torch.full(
        size,
        fill_value,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )


@register_decomposition(aten.empty_like)
@out_wrapper()
def empty_like(
    a: TensorLikeType,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLikeType] = None,
    layout: Optional[torch.layout] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> TensorLikeType:
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    if memory_format != torch.preserve_format:
        return torch.empty(
            a.shape,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
            memory_format=memory_format,
        )

    # memory_format == torch.preserve_format
    logical_to_physical_perm = (
        utils.compute_elementwise_output_logical_to_physical_perm(a)
    )
    # identity perm is [2, 1, 0]
    return torch.empty_permuted(
        a.shape,
        logical_to_physical_perm,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_decomposition([aten.arange.start_step, aten.arange.start_out])
@out_wrapper()
def arange(
    start: NumberType = 0,
    end: Optional[NumberType] = None,
    step: NumberType = 1,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    utils.check_layout(layout)
    utils.check_pin_memory(pin_memory)
    device = torch.device(utils.device_or_default(device))

    assert not isinstance(start, complex)
    assert not isinstance(end, complex)
    assert not isinstance(step, complex)

    # Case: torch.arange(5)
    if end is None:
        end = start
        start = 0
    torch._check(step != 0, lambda: "step must be nonzero")
    if step > 0:
        torch._check(
            end >= start,
            lambda: "upper bound and lower bound inconsistent with step sign",
        )
    elif step < 0:
        torch._check(
            end <= start,
            lambda: "upper bound and lower bound inconsistent with step sign",
        )

    def is_finite(x):
        return not isinstance(x, FloatWithoutSymFloat) or math.isfinite(x)

    torch._check(
        is_finite(start) and is_finite(end),
        lambda: f"unsupported range: {start} -> {end}",
    )
    torch._check(
        is_finite(step),
        lambda: f"step must be finite but got {step}",
    )

    args = (start, end, step)
    integer_args = builtins.all(isinstance(arg, IntLike) for arg in args)

    if dtype is None:
        dtype = torch.int64 if integer_args else torch.get_default_dtype()

    is_integer = utils.is_integer_dtype(dtype)
    if is_integer:
        xstart = sym_int(start)
        xend = sym_int(end)
        xstep = sym_int(step)

    # For int64 we truncate arguments to int before calculating length, but
    # other integral dtypes we don't. Weird... but needed to match ATen shapes.
    if dtype == torch.int64:
        # Uses floordiv to avoid ceil in inductor.
        sgn = bool(xstep > 0) - bool(xstep < 0)  # type: ignore[possibly-undefined]
        length = (xend - xstart + xstep - sgn) // xstep  # type: ignore[possibly-undefined]
    else:
        length = math.ceil((end - start) / step)

    if is_integer:
        return prims.iota(
            length,
            start=xstart,  # type: ignore[possibly-undefined]
            step=xstep,  # type: ignore[possibly-undefined]
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    index = prims.iota(
        length,
        start=0,
        step=1,
        dtype=torch.int64,
        device=device,
        requires_grad=False,
    )

    computation_dtype = (
        torch.long if integer_args else utils.get_acc_type(dtype, device)
    )
    index = _maybe_convert_to_dtype(index, computation_dtype)
    result = start + step * index
    result = _maybe_convert_to_dtype(result, dtype)

    if requires_grad:
        result.requires_grad_(True)
    return result


@register_decomposition(aten.lerp)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("start", "end", "weight"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def lerp(start: Tensor, end: Tensor, weight: Union[Tensor, NumberType]):
    inputs = [start, end]
    if isinstance(weight, Number):
        weight = start.new_full((), weight)  # type: ignore[arg-type]
    else:
        inputs.append(weight)
    assert isinstance(weight, Tensor)  # mypy
    # We implement it this way for numerical stability. We assume (in the stability optimisation)
    # that 0 <= weight <= 1. We take the abs to deal with complex numbers
    # We want to perform operations near zero, which is where floating points are most precise
    # thus, we perform the following optimisation:
    # If weight.abs() >= 0.5:
    #    return (1 - weight) * (start - end) + end
    mask = weight.abs() >= 0.5
    coeff = torch.where(mask, weight - 1, weight)
    base = torch.where(mask, end, start)
    output = coeff * (end - start) + base
    # make sure the decomposition output's stride is same as non-decomposition path.
    stride = utils.compute_elementwise_output_strides(*_maybe_broadcast(*inputs))
    if output.stride() != stride:
        output = prims.copy_strided(output, stride)

    return handle_noncontiguous_outputs(inputs, output)


@register_decomposition(aten.linspace)
@out_wrapper()
def linspace(
    start: Union[NumberType, TensorLikeType],
    end: Union[NumberType, TensorLikeType],
    steps: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLikeType] = None,
    layout: torch.layout = torch.strided,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    if isinstance(start, TensorLikeType):
        torch._check(
            start.dim() == 0,
            lambda: "linspace only supports 0-dimensional start and end tensors",
        )
        start = _maybe_convert_to_dtype(start, torch.float64)
    if isinstance(end, TensorLikeType):
        torch._check(
            end.dim() == 0,
            lambda: "linspace only supports 0-dimensional start and end tensors",
        )
        end = _maybe_convert_to_dtype(end, torch.float64)

    if builtins.any(isinstance(arg, complex) for arg in (start, end, steps)):
        default_complex_dtype = utils.corresponding_complex_dtype(
            torch.get_default_dtype()
        )
        if dtype is None:
            dtype = default_complex_dtype
        else:
            torch._check(
                utils.is_complex_dtype(dtype),
                lambda: f"linspace(): inferred dtype {default_complex_dtype} can't be safely cast to passed dtype {dtype}",
            )
    else:
        dtype = dtype or torch.get_default_dtype()
    assert isinstance(dtype, torch.dtype)

    # steps does not participate in the computation of the dtype
    torch._check_type(
        isinstance(steps, IntLike),
        lambda: f"received an invalid combination of arguments - got \
({type(start).__name__}, {type(end).__name__}, {type(steps).__name__})",
    )
    assert isinstance(steps, IntLike)  # for mypy
    torch._check(steps >= 0, lambda: "number of steps must be non-negative")

    factory_kwargs = {
        "layout": layout,
        "device": device,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
    }
    if steps == 0:
        return torch.full((0,), 0, dtype=dtype, **factory_kwargs)  # type: ignore[arg-type]
    if steps == 1:
        if isinstance(start, TensorLikeType):
            return torch.empty((steps,), dtype=dtype, **factory_kwargs).copy_(start)  # type: ignore[arg-type]
        else:
            return torch.full((steps,), start, dtype=dtype, **factory_kwargs)  # type: ignore[arg-type]

    # Perform in arange in int because some backends like ATen or Triton do not support all the dtypes
    rg = torch.arange(0, steps, **factory_kwargs)  # type: ignore[arg-type]

    # Small types need to be computed in higher precision as this is, at heart, an associative scan
    dtype_red = (
        torch.int64
        if (utils.is_boolean_dtype(dtype) or utils.is_integer_dtype(dtype))
        else dtype
    )
    computation_dtype, _ = utils.reduction_dtypes(
        rg, REDUCTION_OUTPUT_TYPE_KIND.SAME, dtype_red
    )
    cast_rg = partial(_maybe_convert_to_dtype, dtype=computation_dtype)

    # We implement torch.lerp without performing rg / (steps - 1) explicitly
    # With this we get out[0] == start, out[-1] == end
    step = (end - start) / (steps - 1)
    out = torch.where(
        rg < steps / 2,
        start + step * cast_rg(rg),  # type: ignore[arg-type,operator]
        end - step * cast_rg((steps - 1) - rg),  # type: ignore[arg-type,operator]
    )
    return _maybe_convert_to_dtype(out, dtype)  # type: ignore[return-value]


@register_decomposition(aten.logspace)
@out_wrapper()
def logspace(
    start: Union[NumberType, TensorLikeType],
    end: Union[NumberType, TensorLikeType],
    steps: NumberType,
    base: NumberType = 10,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLikeType] = None,
    layout: torch.layout = torch.strided,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    if dtype is None:
        dtype = torch.get_default_dtype()

    # NB: NumPy doesn't have this cast
    if prims.utils.is_integer_dtype(dtype):
        if isinstance(start, FloatLike):
            start = sym_int(start)
        elif isinstance(start, TensorLikeType):
            torch._check(
                start.dim() == 0,
                lambda: "logspace only supports 0-dimensional start and end tensors",
            )
            start = _maybe_convert_to_dtype(start, dtype)
        if isinstance(end, FloatLike):
            end = sym_int(end)
        elif isinstance(end, TensorLikeType):
            torch._check(
                end.dim() == 0,
                lambda: "logspace only supports 0-dimensional start and end tensors",
            )
            end = _maybe_convert_to_dtype(end, dtype)

    if builtins.any(isinstance(arg, complex) for arg in (start, end, steps)):
        default_complex_dtype = utils.corresponding_complex_dtype(
            torch.get_default_dtype()
        )
        dtype = default_complex_dtype
        _dtype = None  # torch.linspace will update the correct dtype
    else:
        _dtype = torch.float64

    assert not isinstance(base, complex)  # for mypy
    if base < 0:
        raise NotImplementedError
    ret = torch.linspace(  # type: ignore[misc]
        start,  # type: ignore[arg-type]
        end,  # type: ignore[arg-type]
        steps,  # type: ignore[arg-type]
        dtype=_dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )
    return _maybe_convert_to_dtype(torch.pow(base, ret), dtype)  # type: ignore[arg-type,return-value]


@overload
def meshgrid(tensors: Sequence[TensorLikeType], indexing: str):
    pass


@overload
def meshgrid(*tensors: TensorLikeType, indexing: str):
    pass


@register_decomposition(aten.meshgrid)  # type: ignore[misc]
def meshgrid(
    *tensors: Union[TensorLikeType, List[TensorLikeType], Tuple[TensorLikeType]],
    indexing: str,
) -> List[TensorLikeType]:
    # This ref simultaneously handles two overloads (see stubs above)
    # The `indexing` argument is currently optional for torch.meshgrid, but we
    # plan to make the argument required: https://github.com/pytorch/pytorch/issues/50276
    if isinstance(tensors[0], (list, tuple)):
        assert len(tensors) == 1
        tensors = tuple(tensors[0])

    torch._check(
        builtins.all(isinstance(a, TensorLike) for a in tensors),
        lambda: "meshgrid expects its inputs to be tensors",
    )

    torch._check(len(tensors) > 0, lambda: "meshgrid expects a non-empty TensorList")

    for i in range(len(tensors) - 1):
        torch._check(
            tensors[i].dtype == tensors[i + 1].dtype,  # type: ignore[union-attr]
            lambda: "meshgrid expects all tensors to have the same dtype",
        )
        torch._check(
            tensors[i].device == tensors[i + 1].device,  # type: ignore[union-attr]
            lambda: "meshgrid expects all tensors to have the same device",
        )

    swap_first_and_second_tensors = False
    if indexing == "xy":
        swap_first_and_second_tensors = len(tensors) >= 2
        if swap_first_and_second_tensors:
            tensors = (tensors[1], tensors[0], *tensors[2:])
    else:
        torch._check(
            indexing == "ij",
            lambda: (
                'torch.meshgrid: indexing must be one of "xy" or "ij", '
                f"but received: {indexing}"
            ),
        )

    result_shape: List[int] = []
    for t in tensors:
        assert isinstance(t, TensorLike)  # mypy
        torch._check(
            t.ndim == 0 or t.ndim == 1,
            lambda: f"torch.meshgrid: Expected 0D or 1D tensor in the tensor list but got: {t}",
        )
        result_shape.append(t.numel())

    grids: List[TensorLikeType] = []
    for i, t in enumerate(tensors):
        assert isinstance(t, TensorLike)  # mypy
        if t.ndim == 0:
            t = t.view((1,))
        grids.append(prims.broadcast_in_dim(t, result_shape, (i,)))

    if swap_first_and_second_tensors:
        # Swap outputs if we originally swapped at the beginning
        grids[0], grids[1] = grids[1], grids[0]

    return grids


# CompositeImplicitAutograd - don't register decomp
def movedim(
    input: TensorLikeType,
    source: Union[int, DimsSequenceType],
    destination: Union[int, DimsSequenceType],
) -> TensorLikeType:
    """
    Reference implementation of torch.movedim
    """
    if type(source) is int:
        source = (source,)
    if type(destination) is int:
        destination = (destination,)

    # Converts to list to produce a compatible error message with core PyTorch,
    # which prints sequences in square brackets.
    torch._check(
        len(source) == len(destination),  # type: ignore[arg-type]
        lambda: (
            "movedim: Invalid source or destination dims: source "  # type: ignore[arg-type]
            f"({list(source)} dims) should contain the same number "  # type: ignore[arg-type]
            f"of dims as destination ({list(destination)} dims)"  # type: ignore[arg-type]
        ),
    )

    rank = input.ndim
    ss = tuple(utils.canonicalize_dims(rank=rank, indices=source))  # type: ignore[arg-type]
    ds = tuple(utils.canonicalize_dims(rank=rank, indices=destination))  # type: ignore[arg-type]

    sss = set(ss)
    dss = set(ds)

    # See above on why this converts to list in error messages.
    torch._check(
        len(ss) == len(sss),
        lambda: f"movedim: repeated dim in `source` ({list(source)})",  # type: ignore[arg-type]
    )
    torch._check(
        len(ds) == len(dss),
        lambda: f"movedim: repeated dim in `destination` ({list(destination)})",  # type: ignore[arg-type]
    )

    m = dict(zip(ds, ss))
    dims = []
    si = 0  # source index
    for di in range(rank):
        # check if the destination index is in the mapping
        s = m.get(di)
        if s is not None:
            # insert source index if found
            dims.append(s)
        else:
            # insert source index sequentially, skipping indices from the mapping
            while si in sss:
                si += 1
            dims.append(si)
            si += 1

    result = torch.permute(input, tuple(dims))

    return result


# NOTE: for convenience, shape can be a tuple of ints or a tuple containing a tuple of ints
@register_decomposition(aten.empty_strided)
@out_wrapper()
def empty_strided(
    shape: Union[ShapeType, Tuple[ShapeType]],
    strides: StrideType,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLikeType] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> TensorLikeType:
    # Layout == strided, pin_memory is False
    utils.check_layout(layout)
    utils.check_pin_memory(pin_memory)

    shape = utils.extract_shape_from_varargs(shape)
    dtype = torch.get_default_dtype() if dtype is None else dtype
    device = torch.device("cpu") if device is None else device

    return prims.empty_strided(
        shape,
        strides,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


@register_decomposition(aten.eye)
@out_wrapper()
def eye(
    n: int,
    m: Optional[int] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,  # TODO: unused
) -> TensorLikeType:
    """
    Reference implementation of torch.eye
    """
    if m is None:
        m = n

    torch._check(n >= 0, lambda: f"n must be greater or equal to 0, got {n}")
    torch._check(m >= 0, lambda: f"m must be greater or equal to 0, got {m}")

    range_n = torch.arange(n, dtype=torch.int64, device=device, requires_grad=False)
    range_m = torch.arange(m, dtype=torch.int64, device=device, requires_grad=False)

    cond = range_n.unsqueeze(-1) == range_m
    if dtype is torch.bool:
        return cond
    else:
        one = torch.ones(
            (1,),
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=False,
        )
        return torch.where(cond, one, 0)
    # TODO: Use requires_grad.  All refs taking the requires_grad kwarg must
    # return a leaf tensor.
    # result.requires_grad_(requires_grad)


@register_decomposition([aten.full.default, aten.full.out])
@out_wrapper()
def full(
    shape: ShapeType,
    fill_value: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    utils.check_layout(layout)
    utils.check_pin_memory(pin_memory)

    dtype = dtype if dtype is not None else utils.type_to_dtype(type(fill_value))
    device = device if device is not None else torch.device("cpu")

    e = empty(
        shape,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )
    return torch.fill(e, fill_value)  # type: ignore[arg-type]


def full_like(
    a: TensorLikeType,
    fill_value: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> TensorLikeType:
    e = torch.empty_like(
        a,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
        memory_format=memory_format,
    )
    return fill(e, fill_value)


@register_decomposition(aten.zeros_like)
@out_wrapper()
def zeros_like(
    a: TensorLikeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> TensorLikeType:
    return torch.full_like(
        a,
        False if (dtype or a.dtype) == torch.bool else 0,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
        memory_format=memory_format,
    )


@register_decomposition(aten.ones_like)
@out_wrapper()
def ones_like(
    a: TensorLikeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> TensorLikeType:
    return torch.full_like(
        a,
        True if (dtype or a.dtype) == torch.bool else 1,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
        memory_format=memory_format,
    )


@register_decomposition(aten.randn.default)
@out_wrapper()
def randn(
    *shape,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLikeType] = None,
    layout: Optional[torch.layout] = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> TensorLikeType:
    utils.check_pin_memory(pin_memory)

    shape_ = utils.extract_shape_from_varargs(shape)

    dtype = utils.dtype_or_default(dtype)
    device = utils.device_or_default(device)

    return prims.normal(
        shape_,
        mean=0.0,
        std=1.0,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def scalar_tensor(
    a: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
) -> TensorLikeType:
    utils.check_layout(layout)
    utils.check_pin_memory(pin_memory)
    dtype = dtype if dtype is not None else utils.type_to_dtype(type(a))
    device = device if device is not None else torch.device("cpu")
    return prims.scalar_tensor(a, dtype=dtype, device=device)


#
# Randomness References
#


def _uniform_helper(
    shape: ShapeType,
    low: Union[bool, int, float] = 0.0,
    high: Union[bool, int, float] = 1.0,
    *,
    dtype: torch.dtype,
    device: DeviceLikeType,
) -> TensorLikeType:
    utils.validate_shape(shape)

    assert isinstance(low, Number)
    assert isinstance(high, Number)
    low = sym_float(low)
    high = sym_float(high)

    assert isinstance(dtype, torch.dtype)
    device = utils.canonicalize_device(device)

    return prims._uniform_helper(shape, low=low, high=high, dtype=dtype, device=device)


@register_decomposition(aten.masked_fill)
@out_wrapper()
def masked_fill(a: TensorLikeType, mask: TensorLikeType, value: TensorOrNumberLikeType):
    python_type = utils.dtype_to_type(a.dtype)
    if isinstance(value, Number):
        value_type = type(value)
    else:
        # NOTE: Could not use value = item(value) as it resulted in
        # RuntimeError: Cannot cast FakeTensor(cpu) to number
        value_ndim = value.ndim
        torch._check(
            value_ndim == 0,
            lambda: f"only supports a 0-dimensional value tensor, but got tensor with {value_ndim} dimension",
        )
        # `masked_fill` allows cpu scalar to be moved to cuda, xpu and hpu but not otherwise.
        is_cpu_scalar = (
            a.device.type
            in ["cuda", "xpu", torch._C._get_privateuse1_backend_name(), "hpu"]
            and value.device.type == "cpu"
        )
        torch._check(
            is_cpu_scalar or value.device == a.device,
            lambda: "Expected `value` to be on same device as `a`",
        )
        value_type = utils.dtype_to_type(value.dtype)

    if value_type is complex:
        # only downcasting from complex to lower type is not allowed.
        # We allow casting `value` to lower type for other case
        # Eg. float -> int.
        # Ref: https://github.com/pytorch/pytorch/issues/79195
        torch._check(
            utils.is_weakly_lesser_type(value_type, python_type),
            lambda: f"could not convert to type {python_type} without overflow",
        )

    # Since `where` allows type-promotion,
    # cast value to correct type before passing to `where`
    value = _maybe_convert_to_dtype(value, a.dtype)
    r = torch.where(mask, value, a)  # type: ignore[arg-type]

    # aten.mask_fill always return a new contiguous tensor
    # contiguous() is needed to correctly model the output stride
    return r.contiguous()


@register_decomposition(aten.masked_fill_)
def masked_fill_(
    a: TensorLikeType, mask: TensorLikeType, value: TensorOrNumberLikeType
) -> TensorLikeType:
    b = torch.masked_fill(a, mask, value)  # type: ignore[arg-type]
    a.copy_(b)
    return a


# CompositeImplicitAutograd - don't register decomp
def allclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """
    Reference implementation of torch.allclose
    """
    _check_close_args(name="torch.allclose", a=a, b=b, rtol=rtol, atol=atol)

    return bool(
        torch.all(torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)).item()
    )


def equal(a: TensorLikeType, b: TensorLikeType) -> bool:
    utils.check_same_device(a, b, allow_cpu_scalar_tensors=False)
    utils.check_same_dtype(a, b)

    # Shape check
    if a.ndim != b.ndim:
        return False

    for x, y in zip(a.shape, b.shape):
        if x != y:
            return False

    # Short-circuits if there are no elements to validate
    if a.numel() == 0:
        return True

    return item(all(eq(a, b)))  # type: ignore[return-value]


@register_decomposition(aten.norm)
@out_wrapper(exact_dtype=True)
def norm(
    input: TensorLikeType,
    p: Optional[Union[float, str]] = "fro",
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # In these cases we compute the "Frobenius norm"
    if (
        p == "fro" and (dim is None or isinstance(dim, Dim) or len(dim) <= 2)
    ) or p is None:
        p = 2
    if isinstance(dim, Dim):
        dim = [dim]
    if isinstance(p, str):
        # Here we either call the nuclear norm, or we call matrix_norm with some arguments
        # that will throw an error
        if dim is None:
            dim = tuple(range(input.ndim))
        return torch.linalg.matrix_norm(input, p, dim, keepdim, dtype=dtype)
    else:
        return torch.linalg.vector_norm(input, p, dim, keepdim, dtype=dtype)


@register_decomposition(aten.trace)
@out_wrapper()
def trace(self: TensorLikeType) -> TensorLikeType:
    torch._check(
        self.ndim == 2, lambda: "expected a matrix, but got tensor with dim {self.ndim}"
    )
    return torch.sum(torch.diag(self, 0))


def _make_r_binary_op(base_op):
    def rop(
        a: Union[TensorLikeType, NumberType],
        b: Union[TensorLikeType, NumberType],
    ) -> TensorLikeType:
        return base_op(b, a)

    return rop


rtruediv = _make_r_binary_op(true_divide)
rfloordiv = _make_r_binary_op(floor_divide)
rpow = _make_r_binary_op(pow)


@register_decomposition(aten.triu)
@out_wrapper()
def triu(a: TensorLikeType, diagonal: int = 0) -> TensorLikeType:
    torch._check(
        a.ndim >= 2, lambda: "triu: input tensor must have at least 2 dimensions"
    )
    h, w = a.shape[-2:]
    mask = (
        torch.arange(w, device=a.device).unsqueeze(-2)
        - torch.arange(h, device=a.device).unsqueeze(-1)
    ) >= diagonal

    # aten.triu always returns a new contiguous tensor
    # contiguous() is needed to correctly model the output stride
    return utils.mask_tensor(mask, a).contiguous()


@register_decomposition(aten.tril)
@out_wrapper()
def tril(a: TensorLikeType, diagonal: int = 0) -> TensorLikeType:
    torch._check(
        a.ndim >= 2, lambda: "tril: input tensor must have at least 2 dimensions"
    )
    h, w = a.shape[-2:]
    mask = (
        torch.arange(w, device=a.device).unsqueeze(-2)
        - torch.arange(h, device=a.device).unsqueeze(-1)
    ) <= diagonal

    # aten.tril always returns a new contiguous tensor
    # contiguous() is needed to correctly model the output stride
    return utils.mask_tensor(mask, a).contiguous()


# This is based on get_tril_size in aten/src/ATen/native/TensorFactories.h
# The components of the matrix that belong to the lower triangle with offset
# form a pentagon that can be broken down into a top trapezoid and a bottom
# rectangle. For the implementation of tril_indices, we need the sizes of
# both of these, as well as the length of the top side of the trapezoid.
def _get_tril_sizes(row: int, col: int, offset: int) -> Tuple[int, int, int]:
    if row == 0 or col == 0:
        return 0, 0, 0

    m_first_row = min(col, 1 + offset) if offset > 0 else int(row + offset > 0)
    m_last_row = max(0, min(col, row + offset))
    n_row_all = max(0, min(row, row + offset))
    n_row_trapezoid = m_last_row - m_first_row + 1

    # Number of elements in top trapezoid
    trapezoid_size = (m_first_row + m_last_row) * n_row_trapezoid // 2
    # Number of elements in bottom rectangle
    diff_row = n_row_all - n_row_trapezoid
    rectangle_size = max(0, diff_row * col)

    return trapezoid_size, rectangle_size, m_first_row


def _trilu_checks(
    name: str,
    row: int,
    col: int,
    dtype: torch.dtype,
    layout: torch.layout,
    pin_memory: bool,
):
    torch._check(row >= 0, lambda: f"row must be non-negative, got {row}")
    torch._check(col >= 0, lambda: f"col must be non-negative, got {col}")
    torch._check(
        dtype in (torch.int32, torch.int64),
        lambda: f"\"{name}\" not implemented for '{dtype}'",
    )


# This is based on tril_indices_cuda in aten/src/ATen/native/cuda/TensorFactories.cu
@register_decomposition(aten.tril_indices)
@out_wrapper()
def tril_indices(
    row: int,
    col: int,
    offset: int = 0,
    *,
    dtype: torch.dtype = torch.long,
    layout: torch.layout = torch.strided,
    device: DeviceLikeType = "cpu",
    pin_memory: bool = False,
) -> TensorLikeType:
    _trilu_checks("tril_indices", row, col, dtype, layout, pin_memory)

    trapezoid_size, rectangle_size, m_first_row = _get_tril_sizes(row, col, offset)
    row_offset = max(0, -offset)

    arange_kw = partial(
        torch.arange, layout=layout, device=device, pin_memory=pin_memory
    )

    # first we do the indices for top trapezoid
    xs1 = arange_kw(0, trapezoid_size, dtype=torch.float64)
    b = m_first_row - 0.5
    row_inds1 = torch.floor(-b + torch.sqrt(b * b + 2 * xs1))
    col_inds1 = torch.floor(xs1 - (2 * m_first_row - 1 + row_inds1) * row_inds1 * 0.5)
    row_inds1 = _maybe_convert_to_dtype(row_inds1 + row_offset, dtype)
    col_inds1 = _maybe_convert_to_dtype(col_inds1, dtype)

    # then bottom rectangle
    xs2 = arange_kw(0, rectangle_size, dtype=dtype)
    row_inds2 = xs2 // col + (col - m_first_row + 1 + row_offset)
    col_inds2 = xs2 % col

    return torch.stack(
        (torch.cat((row_inds1, row_inds2)), torch.cat((col_inds1, col_inds2)))
    )


# Similar to _get_tril_sizes above, but here there is a top trapezoid and
# a bottom rectangle instead. Note that you can't reduce this to
# _get_tril_sizes(col, row, -offset) because that would correspond to
# decomposing into a left trapezoid and right rectangle.
def _get_triu_sizes(row: int, col: int, offset: int) -> Tuple[int, int, int]:
    if row == 0 or col == 0:
        return 0, 0, 0

    m_first_row = max(0, col - offset) if offset > 0 else col

    # Number of elements in top rectangle
    rectangle_size = max(0, min(row, -offset) * col)

    # Number of elements in bottom trapezoid
    trapezoid_size_tril, rectangle_size_tril, _ = _get_tril_sizes(row, col, offset - 1)
    triu_size = row * col - (trapezoid_size_tril + rectangle_size_tril)
    trapezoid_size = triu_size - rectangle_size

    return trapezoid_size, rectangle_size, m_first_row


@register_decomposition(aten.triu_indices)
@out_wrapper()
def triu_indices(
    row: int,
    col: int,
    offset: int = 0,
    *,
    dtype: torch.dtype = torch.long,
    layout: torch.layout = torch.strided,
    device: DeviceLikeType = "cpu",
    pin_memory: bool = False,
) -> TensorLikeType:
    _trilu_checks("triu_indices", row, col, dtype, layout, pin_memory)

    trapezoid_size, rectangle_size, m_first_row = _get_triu_sizes(row, col, offset)
    col_offset = max(0, offset)

    arange_kw = partial(
        torch.arange, layout=layout, device=device, pin_memory=pin_memory
    )

    # indices for top rectangle
    xs2 = arange_kw(0, rectangle_size, dtype=dtype)
    row_inds2 = xs2 // col
    col_inds2 = xs2 % col

    # bottom trapezoid
    xs1 = arange_kw(0, trapezoid_size, dtype=torch.float64)
    b = -0.5 - m_first_row
    row_inds1 = torch.floor(-b - torch.sqrt(b * b - 2 * xs1))
    col_inds1 = torch.floor(xs1 - ((2 * m_first_row - 1 - row_inds1) * row_inds1) * 0.5)
    row_inds1 = _maybe_convert_to_dtype(row_inds1, dtype)
    col_inds1 = _maybe_convert_to_dtype(col_inds1, dtype)

    if col:
        row_inds1 = row_inds1 + (rectangle_size // col)
    col_inds1 = col_inds1 + col_offset

    return torch.stack(
        (torch.cat((row_inds2, row_inds1)), torch.cat((col_inds2, col_inds1)))
    )


@register_decomposition(aten.bucketize)
@out_wrapper(exact_dtype=True)
def bucketize(
    a: TensorLikeType,
    boundaries: TensorLikeType,
    *,
    out_int32: bool = False,
    right: bool = False,
):
    torch._check(
        boundaries.dim() == 1,
        lambda: f"boundaries tensor must be 1 dimension but got dim({boundaries.dim()})",
    )

    out_dtype = torch.int32 if out_int32 else torch.int64
    n_boundaries = boundaries.shape[-1]
    if n_boundaries == 0:
        return torch.zeros_like(a)
    # We are trying to find the bucket (defined by pairs of consecutive elements of `boundaries`)
    # each element of `a` belongs to. We use binary search to achieve logarithimic complexity,
    # but each step of the search is done "in parallel" over all elements of `a`
    # can't use int32 as indexes, so we have to do all computations with int64 and convert at the end
    start = torch.zeros(a.shape, device=a.device, dtype=torch.int64)
    end = start + n_boundaries
    # Max depth of the binary search
    # Since we can't break out of the loop at different points for different elements of a,
    # we just do the max amount of iterations that binary search requires and add condition
    # tensor (cond_update below) to stop updating once the search terminates

    # For first iteration through loop we can skip some checks, we have separate implementation
    mid = start + (end - start) // 2
    mid_val = boundaries[mid]
    if right:
        cond_mid = mid_val > a
    else:
        cond_mid = mid_val >= a
    start = torch.where(cond_mid, start, mid + 1)

    if n_boundaries > 1:
        cond_update = torch.ones_like(a, dtype=torch.bool)
        niters = int(math.log2(n_boundaries))
        for _ in range(niters):
            end = torch.where(cond_mid & cond_update, mid, end)
            cond_update = start < end
            # start might end up pointing to 1 past the end, we guard against that
            mid = torch.where(cond_update, start + (end - start) // 2, 0)
            mid_val = boundaries[mid]
            # If right is true, the buckets are closed on the *left*
            # (i.e., we are doing the equivalent of std::upper_bound in C++)
            # Otherwise they are closed on the right (std::lower_bound)
            if right:
                cond_mid = mid_val > a
            else:
                cond_mid = mid_val >= a
            start = torch.where((~cond_mid) & cond_update, mid + 1, start)

    return start.to(dtype=out_dtype)


@register_decomposition(aten.cauchy)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def cauchy(self, median=0, sigma=1, generator=None):
    assert generator is None
    torch._check(
        not utils.is_complex_dtype(self.dtype)
        and not utils.is_integer_dtype(self.dtype)
        and not utils.is_boolean_dtype(self.dtype),
        lambda: f"Cauchy distribution is a continuous probability distribution. \
        dtype must be a floating point but you specified {self.dtype}",
    )
    torch._check(
        sigma > 0.0,
        lambda: f"cauchy_ expects sigma > 0.0, but found sigma={sigma}",
    )
    return median + sigma * torch.tan(math.pi * (torch.rand_like(self) - 0.5))


@register_decomposition(aten.exponential)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def exponential(self, rate=1, generator=None):
    assert generator is None
    torch._check(
        not utils.is_complex_dtype(self.dtype)
        and not utils.is_integer_dtype(self.dtype)
        and not utils.is_boolean_dtype(self.dtype),
        lambda: f"Exponential distribution is a continuous probability distribution. \
        dtype must be a floating point but you specified {self.dtype}",
    )
    torch._check(
        rate > 0.0,
        lambda: f"exponential_ expects lambda > 0.0, but found lambda={rate}",
    )

    uniform_val = torch.rand_like(self)

    # copying numerics of transformation::exponential see comment:
    # curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
    # we need log to be not 0, and not underflow when converted to half
    # fast __logf approximation can underflow, so set log to -epsilon/2 for 1 or close to 1 args
    epsilon = torch.finfo(uniform_val.dtype).eps / 2
    condition = uniform_val >= 1.0 - epsilon
    log_uniform = torch.where(condition, -epsilon, torch.log(uniform_val))

    return -1 / rate * log_uniform


@register_decomposition(aten.geometric)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def geometric(self, p, generator=None):
    assert generator is None
    # TODO: fix inductor rand_like for integer, bool dtypes
    torch._check(
        not utils.is_complex_dtype(self.dtype)
        and not utils.is_boolean_dtype(self.dtype),
        lambda: f"geometric not implemented for {self.dtype}",
    )
    torch._check(
        0 < p and p < 1,
        lambda: f"geometric_ expects p to be in (0, 1), but got p={p}",
    )
    return torch.floor(torch.log1p(-torch.rand_like(self)) / math.log1p(-p)) + 1


@register_decomposition(aten.log_normal)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def log_normal(self, mean=1, std=2, generator=None):
    assert generator is None
    torch._check(
        not utils.is_complex_dtype(self.dtype)
        and not utils.is_integer_dtype(self.dtype)
        and not utils.is_boolean_dtype(self.dtype),
        lambda: f"log_normal not implemented for {self.dtype}",
    )
    torch._check(
        0 < std,
        lambda: f"log_normal_ expects std > 0.0, but found std={std}",
    )
    return torch.exp(std * torch.randn_like(self) + mean)


# TODO: add support for functionalization aten.normal_functional
# NOTE: the device and dtype will be ignored when shape is None
@register_decomposition(aten.normal)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=(
        "mean",
        "std",
    ),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def normal(
    mean=0,
    std=1,
    size=None,
    *,
    generator=None,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    assert layout is None or layout == torch.strided

    if not isinstance(std, TensorLike):
        torch._check(
            std >= 0, lambda: f"normal expects std >= 0.0, but found std {std}"
        )

    if size is None:
        tensors = tuple(t for t in (mean, std) if isinstance(t, TensorLike))
        torch._check(
            len(tensors) > 0,
            lambda: "normal expects that either mean or std is a tensor, or size is defined",
        )
        torch._check(
            layout is None and pin_memory is None,
            lambda: "Cannot pass layout, or pin_memory without size",
        )

        size = _broadcast_shapes(*(t.shape for t in tensors))
        dtype = tensors[0].dtype
        device = tensors[0].device
    else:
        torch._check(
            not isinstance(mean, TensorLike) and not isinstance(std, TensorLike),
            lambda: "normal expects mean and std to be scalars when size is defined",
        )
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.device("cpu") if device is None else device

    normal_samples = prims.normal(
        size,
        mean=0.0,
        std=1.0,
        dtype=dtype,
        device=device,
        requires_grad=False,
        generator=generator,
    )
    return std * normal_samples + mean


@register_decomposition(aten.normal_)
def normal_(self, mean=0, std=1, *, generator=None):
    return normal(mean, std, self.shape, out=self, generator=generator)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def rad2deg(self: TensorLikeType):
    torch._check(
        not utils.is_complex_dtype(self.dtype),
        lambda: "rad2deg is not supported for complex tensors.",
    )
    M_180_PI = 57.295779513082320876798154814105170332405472466564
    return self * M_180_PI


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def deg2rad(self: TensorLikeType):
    torch._check(
        not utils.is_complex_dtype(self.dtype),
        lambda: "deg2rad is not supported for complex tensors.",
    )
    M_PI_180 = 0.017453292519943295769236907684886127134428718885417
    return self * M_PI_180


@register_decomposition(aten.count_nonzero)
@out_wrapper()
def count_nonzero(self, dim: Optional[DimsType] = None):
    return (self != 0).sum(dim)


def _dot_check(self, other):
    torch._check(
        self.dim() == 1 and other.dim() == 1,
        lambda: f"1D tensors expected, but got {self.dim()}D and {other.dim()}D tensors",
    )

    def numel_error():
        return (
            f"inconsistent tensor size, expected tensor [{self.numel()}] and src [{other.numel()}] to have the"
            f"same number of elements, but got {self.numel()} and {other.numel()} elements respectively"
        )

    torch._check(self.numel() == other.numel(), numel_error)


@register_decomposition(aten.dot)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "other"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def dot(self, other):
    if self.is_complex():
        if self.is_conj():
            if other.is_conj():
                return torch.dot(self.conj(), other.conj()).conj()
            else:
                return torch.vdot(self.conj(), other)
        elif other.is_conj():
            return torch.vdot(other.conj(), self)

    _dot_check(self, other)
    return (self * other).sum()


@register_decomposition(aten.vdot)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "other"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def vdot(self, other):
    if not self.is_complex():
        return torch.dot(self, other)

    if self.is_conj():
        if other.is_conj():
            return torch.vdot(other.conj(), self.conj())
        else:
            return torch.dot(self.conj(), other)
    elif other.is_conj():
        return torch.dot(self, other.conj()).conj()

    _dot_check(self, other)
    # The decomposition fails if you do self.conj()... not sure why
    return (self.conj_physical() * other).sum()


@register_decomposition(aten.select_scatter)
@out_wrapper()
def select_scatter(x: TensorLikeType, src: TensorLikeType, dim: int, index: int):
    dim = utils.canonicalize_dim(x.ndim, dim)
    mask_shape = [1] * x.ndim
    mask_shape[dim] = -1
    if index < 0:
        index = index + x.shape[dim]
    mask = torch.arange(x.shape[dim], device=x.device).view(mask_shape) == index
    src = torch.unsqueeze(src, dim).expand(x.shape)
    return torch.where(mask, src, x)


# inplace
abs_ = _make_inplace(abs)
acos_ = _make_inplace(acos)
acosh_ = _make_inplace(acosh)
add_ = _make_inplace(add)
addcmul_ = _make_inplace(addcmul)
addcdiv_ = _make_inplace(addcdiv)
asin_ = _make_inplace(asin)
asinh_ = _make_inplace(asinh)
atan_ = _make_inplace(atan)
atanh_ = _make_inplace(atanh)
atan2_ = _make_inplace(atan2)
bitwise_and_ = _make_inplace(bitwise_and)
bitwise_left_shift_ = _make_inplace(bitwise_left_shift)
bitwise_not_ = _make_inplace(bitwise_not)
bitwise_or_ = _make_inplace(bitwise_or)
bitwise_right_shift_ = _make_inplace(bitwise_right_shift)
bitwise_xor_ = _make_inplace(bitwise_xor)
ceil_ = _make_inplace(ceil)
clamp_ = _make_inplace(clamp)
clamp_min_ = _make_inplace(clamp_min)
clamp_max_ = _make_inplace(clamp_max)
conj_physical_ = _make_inplace(conj_physical)
copysign_ = _make_inplace(copysign)
cos_ = _make_inplace(cos)
cosh_ = _make_inplace(cosh)
cumsum_ = _make_inplace(cumsum)
cumprod_ = _make_inplace(cumprod)
deg2rad_ = _make_inplace(deg2rad)
digamma_ = _make_inplace(digamma)
div_ = _make_inplace(div)
eq_ = _make_inplace(eq)
erf_ = _make_inplace(erf)
erfc_ = _make_inplace(erfc)
erfinv_ = _make_inplace(erfinv)
exp_ = _make_inplace(exp)
exp2_ = _make_inplace(exp2)
expm1_ = _make_inplace(expm1)
float_power_ = _make_inplace(float_power)
floor_ = _make_inplace(floor)
floor_divide_ = _make_inplace(floor_divide)
fmod_ = _make_inplace(fmod)
frac_ = _make_inplace(frac)
gcd_ = _make_inplace(gcd)
ge_ = _make_inplace(ge)
gt_ = _make_inplace(gt)
heaviside_ = _make_inplace(heaviside)
hypot_ = _make_inplace(hypot)
igamma_ = _make_inplace(igamma)
igammac_ = _make_inplace(igammac)
i0_ = _make_inplace(i0)
lcm_ = _make_inplace(lcm)
le_ = _make_inplace(le)
lerp_ = _make_inplace(lerp)
lgamma_ = _make_inplace(lgamma)
log10_ = _make_inplace(log10)
log1p_ = _make_inplace(log1p)
log2_ = _make_inplace(log2)
log_ = _make_inplace(log)
logical_and_ = _make_inplace(logical_and)
logical_not_ = _make_inplace(logical_not)
logical_or_ = _make_inplace(logical_or)
logical_xor_ = _make_inplace(logical_xor)
lt_ = _make_inplace(lt)
mul_ = _make_inplace(mul)
mvlgamma_ = _make_inplace(mvlgamma)
nan_to_num_ = _make_inplace(nan_to_num)
ne_ = _make_inplace(ne)
neg_ = _make_inplace(neg)
nextafter_ = _make_inplace(nextafter)
pow_ = _make_inplace(pow)
rad2deg_ = _make_inplace(rad2deg)
reciprocal_ = _make_inplace(reciprocal)
remainder_ = _make_inplace(remainder)
rsqrt_ = _make_inplace(rsqrt)
sgn_ = _make_inplace(sgn)
sigmoid_ = _make_inplace(sigmoid)
sign_ = _make_inplace(sign)
sin_ = _make_inplace(sin)
sinc_ = _make_inplace(sinc)
sinh_ = _make_inplace(sinh)
sqrt_ = _make_inplace(sqrt)
square_ = _make_inplace(square)
sub_ = _make_inplace(sub)
tan_ = _make_inplace(tan)
tanh_ = _make_inplace(tanh)
tril_ = _make_inplace(tril)
triu_ = _make_inplace(triu)
true_divide_ = _make_inplace(true_divide)
trunc_ = _make_inplace(trunc)
xlogy_ = _make_inplace(xlogy)
cauchy_ = _make_inplace(cauchy)
exponential_ = _make_inplace(exponential)
geometric_ = _make_inplace(geometric)
log_normal_ = _make_inplace(log_normal)
zero_ = _make_inplace(zero)

alias_copy = _make_copy_from_view(aten.alias)
as_strided_copy = _make_copy_from_view(aten.as_strided)
diagonal_copy = _make_copy_from_view(aten.diagonal)
expand_copy = _make_copy_from_view(aten.expand)
# TODO: This must return a sparse tensor if the input is sparse, but refs have
# no sparse support. See narrow_copy_sparse in core.
narrow_copy = _make_copy_from_view(aten.narrow)
t_copy = _make_copy_from_view(aten.t)
unsqueeze_copy = _make_copy_from_view(aten.unsqueeze)
view_copy = _make_copy_from_view(aten.view)


# xref: isStorage in torch/csrc/DynamicTypes.cpp
def _isStorage(obj):
    return isinstance(obj, (torch.TypedStorage, torch.UntypedStorage))


# xref: compute_sizes in torch/csrc/utils/tensor_new.cpp
def _compute_sizes(seq, scalar_type):
    MAX_DIMS = 128
    is_storage = _isStorage(seq)
    sizes = []
    # TODO: this is inaccurate, we actually test PySequence_Check
    while isinstance(seq, (list, tuple)):
        length = len(seq)
        if is_storage:
            length //= scalar_type.itemsize
        sizes.append(length)
        if len(sizes) > MAX_DIMS:
            raise ValueError(f"too many dimensions '{type(seq).__name__}'")
        if length == 0:
            break
        try:
            handle = seq[0]
        except Exception:
            raise ValueError(  # noqa: B904
                f"could not determine the shape of object type '{type(seq).__name__}'"
            )
        seq = handle

    return sizes


# xref: infer_scalar_type in torch/csrc/utils/tensor_new.cpp
def _infer_scalar_type(obj):
    if isinstance(obj, FloatLike):
        return torch.get_default_dtype()
    if isinstance(obj, IntLike) and not isinstance(obj, bool):  # careful!
        return torch.int64
    if isinstance(obj, BoolLike):
        return torch.bool
    if isinstance(obj, complex):
        default_dtype = torch.get_default_dtype()
        if default_dtype is torch.float:
            return torch.cfloat
        elif default_dtype is torch.double:
            return torch.cdouble
        elif default_dtype is torch.half:
            return torch.chalf
        else:
            raise RuntimeError("invalid default scalar type for complex")
    if isinstance(obj, torch.Tensor):
        return obj.dtype
    if isinstance(obj, str):
        raise TypeError(f"new(): invalid data type '{type(obj).__name__}'")
    # TODO: this is inaccurate, we actually test PySequence_Check
    if isinstance(obj, (list, tuple)):
        scalarType = None
        length = len(obj)
        # match NumPy semantics, except use default tensor type instead of
        # double.
        if length == 0:
            return torch.get_default_dtype()
        for i in range(length):
            cur_item = obj[i]
            # TODO: test this
            """
            if cur_item is obj:
                raise TypeError("new(): self-referential lists are incompatible")
            """
            item_scalarType = _infer_scalar_type(cur_item)  # recurse!
            if scalarType is not None:
                scalarType = torch.promote_types(scalarType, item_scalarType)
            else:
                scalarType = item_scalarType
            if scalarType is torch.cdouble:
                # this won't change (unless we hit undefined, but that will
                # fail later)
                return scalarType
        return scalarType
    raise RuntimeError(f"Could not infer dtype of {type(obj).__name__}")


# Analogous to recursive_store
# xref: recursive_store in torch/csrc/utils/tensor_new.cpp
def _recursive_build(
    scalarType: torch.dtype, obj: Union[TensorOrNumberLikeType, TensorSequenceType]
):
    if isinstance(obj, Tensor) and obj.numel() == 1:
        return obj.detach().to(dtype=scalarType, device="cpu", copy=True).view(())
    elif isinstance(obj, Tensor):
        # It is invalid to call ".tensor([...])" with a non-scalar tensor in eager mode
        # >>> torch.tensor([torch.randn(2)])
        # ValueError: only one element tensors can be converted to Python scalars
        #
        # But it is possible with a NumPy array
        # >>> torch.tensor([np.random.uniform(size=(2,))]).shape
        # torch.Size([1, 2])
        return obj.detach().to(dtype=scalarType, device="cpu", copy=True)
    elif isinstance(obj, Number):
        return torch.scalar_tensor(obj, dtype=scalarType)

    # seq can be a list of tensors
    seq = obj
    return torch.stack([_recursive_build(scalarType, item) for item in seq])


# xref: internal_new_from_data in torch/csrc/utils/tensor_new.cpp
def _internal_new_from_data(
    options,
    scalar_type,
    device_opt,
    data,
    copy_variables,
    copy_numpy,
    type_inference,
    pin_memory=False,
):
    if isinstance(data, torch.Tensor):
        torch._check(
            not pin_memory, lambda: "Can't pin tensor constructed from a variable"
        )
        var = data
        if copy_variables:
            var = var.detach()
        inferred_scalar_type = var.dtype if type_inference else scalar_type
        device = device_opt if device_opt is not None else var.device
        return var.to(
            device=device,
            dtype=inferred_scalar_type,
            non_blocking=False,
            copy=copy_variables,
        )

    # TODO
    if hasattr(data, "__cuda_array_interface__"):
        return NotImplemented

    # TODO: test for numpy input with PyArray_Check

    device = device_opt if device_opt is not None else options["device"]
    inferred_scalar_type = _infer_scalar_type(data) if type_inference else scalar_type

    # NB: Don't need to avoid tracing, as we aren't going to do any manual
    # pointer filling tricks
    if _isStorage(data):
        return NotImplemented
    else:
        if torch.device(device).type == "meta":
            return NotImplemented

        # In the C implementation, we would directly start poking the memory
        # of a freshly allocated CPU tensor.  Here, we're going to do an
        # alternate, heinously slow implementation: turn each individual
        # scalar into a tensor, and then repeatedly cat them together
        tensor = _recursive_build(inferred_scalar_type, data)

        tensor = tensor.to(device, inferred_scalar_type, non_blocking=False, copy=False)

    # NB: lift_fresh is not needed, because we built the tensor from scalars
    # guaranteeing a fresh tensor in this case
    return tensor


# xref: tensor_ctor in torch/csrc/utils/tensor_new.cpp
def tensor(data, *, dtype=None, device=None, pin_memory=False, requires_grad=False):
    # TODO (or not): support names kwarg
    if isinstance(data, torch.Tensor):
        warnings.warn(
            "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
            "or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)"
        )
    type_inference = dtype is None
    new_tensor = _internal_new_from_data(
        # device="cpu" because that's what you get with torch.tensor(2) no
        # device by default
        {"device": "cpu"},  # TODO: use torch.get_default_tensor_type
        dtype if dtype is not None else torch.get_default_dtype(),
        device,
        data,
        copy_variables=True,
        copy_numpy=True,
        type_inference=type_inference,
        pin_memory=pin_memory,
    )
    new_tensor.detach_()
    if requires_grad:
        new_tensor.requires_grad_(requires_grad)
    return new_tensor


# Views
# We can't model these as above, as the pattern of doing `op(a, out=a)` does not work for a view function
# given that it does not reshape the input (it just copies the result into it)

# squeeze_ = _make_inplace(squeeze)
# t_ = _make_inplace(t)
# transpose_ = _make_inplace(transpose)
# unsqueeze_ = _make_inplace(unsqueeze)


import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
