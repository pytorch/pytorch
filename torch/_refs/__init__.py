import builtins
import collections
import math
import operator
import warnings

from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Callable, List, Optional, overload, Sequence, Tuple, Union

import torch

import torch._prims as prims
import torch._prims_common as utils
from torch._prims_common import (
    check,
    DeviceLikeType,
    Dim,
    DimsSequenceType,
    DimsType,
    dtype_to_type,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    FloatLike,
    IntLike,
    is_weakly_lesser_type,
    Number,
    NumberType,
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
from torch.fx.experimental.symbolic_shapes import sym_float, sym_int

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
    "digamma",
    "erf",
    "erfinv",
    "erfc",
    "exp",
    "expm1",
    "exp2",
    "fill",
    "floor",
    "frac",
    "index_add",
    "index_copy",
    "index_copy_",
    "index_select",
    "index_fill",
    "index_fill_",
    "isfinite",
    "isinf",
    "isnan",
    "isreal",
    "i0",
    "lerp",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "log_softmax",
    "nan_to_num",
    "neg",
    "positive",
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
    # "complex",
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
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
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
    "where",
    #
    # Data conversion and movement references
    #
    "clone",
    "copy_to",  # TODO: add OpInfo (or implement .to)
    "item",  # TODO: add OpInfo
    "to",
    #
    # Reduction ops
    #
    "all",
    "amax",
    "amin",
    "any",
    "mean",
    "std_mean",
    "var_mean",
    "sum",
    "sum_to_size",
    "prod",
    "var",
    #
    # Linear algebra ops
    #
    "addr",
    #
    # View & Shape Ops
    #
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "as_strided",
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
    "roll",
    "rot90",
    "rsqrt",
    "stack",
    "swap_axes",  # alias for transpose
    "squeeze",
    "t",
    "T",
    "tensor_split",
    "transpose",
    "unfold",
    "unfold_copy",
    "unsqueeze",
    "view",
    "vsplit",
    "vstack",
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
    "empty",
    "empty_like",
    "empty_strided",
    "eye",
    "full",
    "full_like",
    "linspace",
    "logspace",
    "ones",
    "ones_like",
    "randn",
    "scalar_tensor",
    "zeros",
    "zeros_like",
    #
    # Randomness References
    #
    "uniform",  # TODO: add OpInfo -- and testing for randomness?
    #
    # Test-related functions
    #
    "allclose",
    "equal",  # TODO: add OpInfo
    #
    # Statistical operations
    #
    "bucketize",
]

Tensor = torch.Tensor
DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]


def _broadcast_shapes(*_shapes):
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
    for shape in shapes:
        for idx in range(-1, -1 - len(shape), -1):
            if common_shape[idx] == 1:
                if shape[idx] < 0:
                    raise ValueError(
                        "Attempting to broadcast a dimension with negative length!"
                    )
                common_shape[idx] = shape[idx]
            elif shape[idx] != 1:
                if common_shape[idx] != shape[idx]:
                    raise RuntimeError(
                        "Attempting to broadcast a dimension of length ",
                        str(shape[idx]),
                        "!",
                    )

    return common_shape


def _maybe_broadcast(*args, preserve_cpu_scalar_tensors=True):
    # Computes common shape
    common_shape = _broadcast_shapes(
        *map(lambda t: t.shape if isinstance(t, TensorLike) else None, args)
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
            if not isinstance(a, TensorLike):
                raise RuntimeError(
                    "Expected a tensor input for an elementwise unary operation!"
                )

            if extra_meta is not None:
                extra_meta(a)

            return prim(a)

        if aten_op is infer_aten_op:
            aten_op = getattr(torch.ops.aten, prim.__name__)
        if aten_op is not None:
            register_decomposition(aten_op)(_ref)

        return _ref

    return inner


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


@register_decomposition(torch.ops.aten.conj_physical)
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
        msg = "value argument of type {0} cannot be safely cast to type {1}!".format(
            type(value), python_type
        )
        raise ValueError(msg)

    return prims.fill(a, value)


def fill_(a: TensorLikeType, value: NumberType) -> TensorLikeType:
    r = prims.fill(a, value)
    prims.copy_to(a, r)
    return a


def zero_(a: TensorLikeType) -> TensorLikeType:
    r = prims.fill(a, 0)
    prims.copy_to(a, r)
    return a


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def floor(a):
    return prims.floor(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def frac(x: TensorLikeType) -> TensorLikeType:
    trunc_x = mul(floor(abs(x)), sign(x))
    return sub(x, trunc_x)


# imag does not use _make_elementwise_unary_reference because it does not support out
def imag(a: TensorLikeType) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    utils.check(
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
        return logical_or(isinf(real(a)), isinf(imag(a)))
    return logical_not(logical_or(isnan(a), isfinite(a)))


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def isposinf(a: TensorLikeType) -> TensorLikeType:
    utils.check(
        not utils.is_complex_dtype(a.dtype),
        lambda: f"Complex dtype is not supported for isposinf, got dtype {a.dtype}",
    )
    if utils.is_float_dtype(a.dtype):
        return eq(a, float("inf"))
    return zeros_like(a, dtype=torch.bool)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def isneginf(a: TensorLikeType) -> TensorLikeType:
    utils.check(
        not utils.is_complex_dtype(a.dtype),
        lambda: f"Complex dtype is not supported for isneginf, got dtype {a.dtype}",
    )
    if utils.is_float_dtype(a.dtype):
        return eq(a, float("-inf"))
    return zeros_like(a, dtype=torch.bool)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def isnan(a: TensorLikeType) -> TensorLikeType:
    return prims.ne(a, a)


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
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, aten_op=torch.ops.aten.special_i0
)
def i0(a):
    return prims.bessel_i0(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def lgamma(a):
    return prims.lgamma(a)


# alias
mvlgamma = torch.special.multigammaln  # type: ignore[has-type]


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


@out_wrapper()
def logsumexp(
    a: TensorLikeType,
    dim: DimsType,
    keepdim: bool = False,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(a.ndim, dim)
    # ATen specifies int[1] type dims which expands integers to tuples of length 1
    if not isinstance(dim, Iterable):
        dim = (dim,)
    if utils.is_float_dtype(a.dtype) or utils.is_complex_dtype(a.dtype):
        # For float and complex dtypes, we shift input to exp by a constant to avoid overflow
        a_max = amax(a, dim, keepdim=True)
        a_max = where(abs(a_max) == float("inf"), 0.0, a_max)
        a_max_squeezed = prims.squeeze(a_max, dim) if not keepdim else a_max
        result = log(sum(exp(a - a_max), dim, keepdim=keepdim)) + a_max_squeezed
    else:
        # This case covers boolean and integer dtypes and we use non-stabilized computation
        result = log(sum(exp(a), dim, keepdim=keepdim))
    return result


@register_decomposition(torch.ops.aten.nan_to_num)
@out_wrapper()
def nan_to_num(
    a: TensorLikeType,
    nan: Optional[NumberType] = 0.0,
    posinf: Optional[NumberType] = None,
    neginf: Optional[NumberType] = None,
) -> TensorLikeType:
    assert isinstance(a, TensorLike)

    if utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
        return clone(a)

    if nan is None:
        nan = 0.0

    if posinf is None:
        posinf = prims.maximum_value(a.dtype)

    if neginf is None:
        neginf = prims.minimum_value(a.dtype)

    result = where(isnan(a), nan, a)

    is_neg = signbit(a)
    is_neginf = bitwise_and(isinf(a), is_neg)
    result = where(is_neginf, neginf, result)

    is_posinf = bitwise_and(isinf(a), bitwise_not(is_neg))
    result = where(is_posinf, posinf, result)
    return result


def _neg_meta(a: TensorLikeType):
    if a.dtype is torch.bool:
        msg = "neg is not supported on bool tensors."
        raise RuntimeError(msg)


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


# TODO: round takes additional kwargs
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=None,  # TODO: this does need a decomp, but kwarg handling is needed
)
def round(a):
    return prims.round(a)


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


def _make_elementwise_binary_reference(
    prim: Callable,
    *,
    type_promotion_kind,
    aten_op=infer_aten_op,
    has_out=True,
    supports_lhs_python_scalar=True,
    supports_rhs_python_scalar=True,
    supports_two_python_scalars=False,
) -> Callable:
    @elementwise_type_promotion_wrapper(
        type_promoting_args=("a", "b"),
        type_promotion_kind=type_promotion_kind,
    )
    def _ref(
        a: Union[Tensor, NumberType],
        b: Union[Tensor, NumberType],
    ) -> Tensor:
        if not supports_lhs_python_scalar and isinstance(a, Number):
            raise ValueError(
                "Received a lhs Python scalar to an elementwise binary operation that does not accept lhs scalars!"
            )

        if not supports_rhs_python_scalar and isinstance(b, Number):
            raise ValueError(
                "Received a rhs Python scalar to an elementwise binary operation that does not accept rhs scalars!"
            )

        if (
            not supports_two_python_scalars
            and isinstance(a, Number)
            and isinstance(b, Number)
        ):
            raise ValueError(
                f"Receive two Number inputs to an elementwise binary operation {prim}!"
            )

        a, b = _maybe_broadcast(a, b)
        return prim(a, b)

    if has_out:
        _ref = out_wrapper()(_ref)

    if aten_op is infer_aten_op:
        aten_op = getattr(torch.ops.aten, prim.__name__.split(".")[0])
    if aten_op is not None:
        register_decomposition(aten_op)(_ref)

    return _ref


# Add has its own implementation because it has an alpha argument
@register_decomposition(torch.ops.aten.add)
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
            msg = (
                "alpha argument of type {0} cannot be safely cast to type {1}!".format(
                    type(alpha), python_type
                )
            )
            raise ValueError(msg)
        b = prims.mul(b, alpha)

    return prims.add(a, b)


# TODO: add docstring
atan2 = _make_elementwise_binary_reference(
    prims.atan2,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)

# TODO: add docstring
bitwise_and = _make_elementwise_binary_reference(
    prims.bitwise_and,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: add docstring
bitwise_left_shift = _make_elementwise_binary_reference(
    prims.shift_left,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.bitwise_left_shift,  # prim/aten name mismatch
)

# TODO: add docstring
bitwise_or = _make_elementwise_binary_reference(
    prims.bitwise_or,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: add docstring
bitwise_right_shift = _make_elementwise_binary_reference(
    prims.shift_right_arithmetic,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.bitwise_right_shift,  # prim/aten name mismatch
)

# TODO: add docstring
bitwise_xor = _make_elementwise_binary_reference(
    prims.bitwise_xor,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)


def _copysign(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    if isinstance(b, Number) and isinstance(a, Tensor):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(a, Tensor) and isinstance(b, Tensor) and a.device != b.device:
        msg = "Expected divisor (b) to be on the same device ({0}) as dividend (a), but it is found on {1}!".format(
            a.device, b.device
        )
        raise RuntimeError(msg)
    return where(signbit(b), neg(abs(a)), abs(a))


# TODO: add docstring
copysign = _make_elementwise_binary_reference(
    _copysign,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    aten_op=torch.ops.aten.copysign,
)

# TODO: add docstring
# complex =  _make_elementwise_binary_reference(prims.complex, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)


@register_decomposition(torch.ops.aten.div)
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
        msg = (
            "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
            "but found {0}.".format(rounding_mode)
        )
        raise ValueError(msg)


# TODO: add docstring
eq = _make_elementwise_binary_reference(
    prims.eq,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)


def _pow(
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
    return prims.pow(a, b)


# TODO: add docstring
pow = _make_elementwise_binary_reference(
    _pow,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG,
    aten_op=torch.ops.aten.pow,
)

# TODO: add docstring
# Float power has its own implementation because it has unique type promotion.
# NB: aten_op not registered because CompositeExplicitAutograd
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
    if isinstance(a, TensorLike) and a.dtype != dtype:
        a = prims.to_dtype(a, dtype)
    if isinstance(b, TensorLike) and b.dtype != dtype:
        b = prims.to_dtype(b, dtype)

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
def _floor_divide(
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
            msg = "Expected divisor (b) to be on the same device ({0}) as dividend (a), but it is found on {1}!".format(
                a.device, b.device
            )
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
        check(False, lambda: f"{dtype} not supported for floor_divide")


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


# TODO: add docstring
floor_divide = _make_elementwise_binary_reference(
    _floor_divide,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.floor_divide,
    supports_two_python_scalars=True,
)


# TODO: add docstring
fmax = _make_elementwise_binary_reference(
    prims.fmax,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.fmax,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)

# TODO: add docstring
fmin = _make_elementwise_binary_reference(
    prims.fmin,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.fmin,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)

# TODO: add docstring
fmod = _make_elementwise_binary_reference(
    prims.fmod,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.fmod,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=True,
)

# TODO: add docstring
gcd = _make_elementwise_binary_reference(
    prims.gcd,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.gcd,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)

# TODO: add docstring
ge = _make_elementwise_binary_reference(
    prims.ge,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)

# TODO: add docstring
gt = _make_elementwise_binary_reference(
    prims.gt,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)


def _heaviside(input: TensorLikeType, values: TensorLikeType) -> TensorLikeType:
    input_eq_zero = eq(input, 0)
    input_lt_zero = logical_or(lt(input, 0), isnan(input))
    zeros_and_ones = where(input_lt_zero, 0, 1)
    output = where(input_eq_zero, values, zeros_and_ones)
    return output


heaviside = _make_elementwise_binary_reference(
    _heaviside,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
    aten_op=torch.ops.aten.heaviside,
)

hypot = _make_elementwise_binary_reference(
    prims.hypot,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)

igamma = _make_elementwise_binary_reference(
    prims.igamma,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)

igammac = _make_elementwise_binary_reference(
    prims.igammac,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)


def _check_close_args(
    name: str,
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float,
    atol: float,
) -> None:
    check(
        a.dtype == b.dtype,
        lambda: "{0}: Attempting to compare tensors of different dtypes {1} and {2}!".format(
            name, a.dtype, b.dtype
        ),
        ValueError,
    )
    check(
        rtol >= 0,
        lambda: "{0}: rtol must be greater than or equal to zero, but got {1}!".format(
            name, rtol
        ),
    )
    check(
        atol >= 0,
        lambda: "{0}: atol must be greater than or equal to zero, but got {1}!".format(
            name, atol
        ),
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


def _lcm(a: TensorLikeType, b: TensorLikeType):
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


# TODO: add docstring
lcm = _make_elementwise_binary_reference(
    _lcm,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.lcm,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)


# TODO: add docstring
le = _make_elementwise_binary_reference(
    prims.le,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)


def _logical_and(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        a = a != 0
    if not utils.is_boolean_dtype(b.dtype):
        b = b != 0
    return a & b


logical_and = _make_elementwise_binary_reference(
    _logical_and,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=torch.ops.aten.logical_and,
)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL, aten_op=torch.ops.aten.logical_not
)
def logical_not(a: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        return a == 0
    return ~a


def _logical_or(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        a = a != 0
    if not utils.is_boolean_dtype(b.dtype):
        b = b != 0
    return bitwise_or(a, b)


logical_or = _make_elementwise_binary_reference(
    _logical_or,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=torch.ops.aten.logical_or,
)


def _logical_xor(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        a = a != 0
    if not utils.is_boolean_dtype(b.dtype):
        b = b != 0
    return a ^ b


# TODO: skip unnecessary conversion of long to float
logical_xor = _make_elementwise_binary_reference(
    _logical_xor,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=torch.ops.aten.logical_xor,
)


# TODO: add docstring
lt = _make_elementwise_binary_reference(
    prims.lt,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)

# TODO: add docstring
maximum = _make_elementwise_binary_reference(
    prims.maximum,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: add docstring
minimum = _make_elementwise_binary_reference(
    prims.minimum,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: add docstring
mul = _make_elementwise_binary_reference(
    prims.mul,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_two_python_scalars=True,
)

# TODO: add docstring
ne = _make_elementwise_binary_reference(
    prims.ne,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)

# TODO: add docstring
nextafter = _make_elementwise_binary_reference(
    prims.nextafter,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)

# TODO: add docstring
remainder = _make_elementwise_binary_reference(
    prims.remainder,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.remainder,
)

# reverse sub
def rsub(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    alpha: Optional[NumberType] = None,
):
    if isinstance(a, Number):
        msg = "Received a Number for the first argument, but expected a Tensor"
        raise ValueError(msg)
    return sub(b, a, alpha=alpha)


# TODO: add docstring
# TODO: consider refactoring this with add impl
# sub has its own implementation because it has an alpha argument
@register_decomposition(torch.ops.aten.sub)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def sub(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    alpha: Optional[NumberType] = None,
):
    """
    Reference implementation of torch.sub
    """

    a, b = _maybe_broadcast(a, b)

    if alpha is not None:
        dtype = a.dtype if isinstance(a, TensorLike) else b.dtype  # type: ignore[union-attr]
        python_type = utils.dtype_to_type(dtype)
        if not utils.is_weakly_lesser_type(type(alpha), python_type):
            msg = (
                "alpha argument of type {0} cannot be safely cast to type {1}!".format(
                    type(alpha), python_type
                )
            )
            raise ValueError(msg)
        b = prims.mul(b, alpha)

    return prims.sub(a, b)


# TODO: add docstring
true_divide = _make_elementwise_binary_reference(
    prims.div,  # type: ignore[has-type]
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=None,  # CompositeImplicitAutograd
    supports_two_python_scalars=True,
)


@register_decomposition(torch.ops.aten.xlogy)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def xlogy(a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]):
    utils.check(
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


def _trunc_divide(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    dtype = utils.get_dtype(a)
    if utils.is_integer_dtype(dtype):
        return prims.div(a, b)

    return trunc(prims.div(a, b))


# TODO: add docstring
trunc_divide = _make_elementwise_binary_reference(
    _trunc_divide,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=None,  # CompositeImplicitAutograd
    supports_two_python_scalars=True,
)

#
# Elementwise Ternary References
#


@register_decomposition(torch.ops.aten.addcdiv)
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
        check(
            utils.is_weakly_lesser_type(type(value), python_type),
            lambda: "value argument of type {0} cannot be safely cast to type {1}!".format(
                type(value), python_type
            ),
            exc_type=ValueError,
        )

    return self + value * tensor1 / tensor2


@register_decomposition(torch.ops.aten.addcmul)
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
        check(
            utils.is_weakly_lesser_type(type(value), python_type),
            lambda: "value argument of type {0} cannot be safely cast to type {1}!".format(
                type(value), python_type
            ),
            exc_type=ValueError,
        )

    return self + value * tensor1 * tensor2


@register_decomposition(torch.ops.aten.clamp)
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


@register_decomposition(torch.ops.aten.clamp_min)
@out_wrapper()
def clamp_min(
    self: TensorLikeType,
    min: TensorOrNumberLikeType = None,
) -> TensorLikeType:
    return torch.clamp(self, min=min)  # type: ignore[arg-type]


@register_decomposition(torch.ops.aten.clamp_max)
@out_wrapper()
def clamp_max(
    self: TensorLikeType,
    max: TensorOrNumberLikeType = None,
) -> TensorLikeType:
    return torch.clamp(self, max=max)  # type: ignore[arg-type]


#
# Conditional references
#

# https://pytorch.org/docs/stable/generated/torch.where.html
# TODO: implement alternate where
@register_decomposition(torch.ops.aten.where)
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
    check(
        pred.dtype is torch.bool,
        lambda: f"expected predicate to be bool, got {pred.dtype}",
    )

    pred, a, b = _maybe_broadcast(pred, a, b)
    return prims.where(pred, a, b)


#
# Data Movement References
#
@register_decomposition(torch.ops.aten.clone)
def clone(
    a: TensorLikeType, *, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    result = prims.clone(a, memory_format=memory_format)
    return result


def copy_to(a: Tensor, b: Tensor, *, allow_cross_device=True):
    if not allow_cross_device and a.device != b.device:
        msg = "Attempting to copy from device {0} to device {1}, but cross-device copies are not allowed!".format(
            b.device, a.device
        )
        raise RuntimeError(msg)

    return prims.copy_to(a, b)


@register_decomposition(torch.ops.aten.item)
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
    device: Optional[torch.device] = None,
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
):
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
):
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
):
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
):
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
def canonicalize_to_arguments(a: Tensor, to_kwargs: dict):
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
    canonicalize_to_arguments(a, kwargs)

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
            "Received a tensor with {0} dimensions, but only tensors with up to 64 dims are supported!".format(
                a.ndim
            )
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
        valid_shape = a.ndim == 0 or py_all(a.shape[i] for i in dims)
        if not valid_shape:
            raise RuntimeError(
                "reducing over zero-size dimension for reduction operation without identity"
            )
    computation_dtype, result_dtype = utils.reduction_dtypes(
        a, output_dtype_kind, dtype
    )
    a = _maybe_convert_to_dtype(a, computation_dtype)  # type: ignore[assignment]
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
    name = fn.__name__
    fn = out_wrapper()(fn)

    def _fn(*args, out=None, **kwargs):
        result = fn(*args, out=out, **kwargs)
        if out is None:
            return result.clone(memory_format=torch.contiguous_format)
        return result

    copy_name = f"{name}_copy"
    _fn.__name__ = copy_name
    _fn = register_decomposition(getattr(torch.ops.aten, copy_name))(_fn)
    return _fn


# Saves Python all
py_all = all


@register_decomposition(torch.ops.aten.all)
@out_wrapper()
def all(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
) -> TensorLikeType:
    # Computes nelem
    if isinstance(dim, Dim):
        dim = (dim,)  # type: ignore[assignment]

    a_ = _maybe_convert_to_dtype(a, torch.bool)
    # avoid comparison with symbolic number of elements to make this op symint friendly
    result = eq(sum(logical_not(a_), dim=dim, keepdim=keepdim), 0)

    # Preserves uint8 -- probably a legacy mask thing
    if a.dtype is torch.uint8:
        return prims.convert_element_type(result, torch.uint8)

    return result


# Saves Python any
py_any = any


@register_decomposition(torch.ops.aten.any)
@out_wrapper()
def any(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
) -> TensorLikeType:
    a_ = _maybe_convert_to_dtype(a, torch.bool)
    result = ne(sum(a_, dim=dim, keepdim=keepdim), False)  # type: ignore[arg-type]

    # Preserves uint8 -- probably a legacy mask thing
    if a.dtype is torch.uint8:
        return prims.convert_element_type(result, torch.uint8)

    return result


@register_decomposition(torch.ops.aten.sum)
def sum(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    if dtype is None:
        if utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
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
    utils.check(
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


@register_decomposition(torch.ops.aten.prod)
def prod(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    dtype=None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    if dtype is None:
        if utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
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


@register_decomposition(torch.ops.aten.amin)
def amin(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
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


@register_decomposition(torch.ops.aten.amax)
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


@register_decomposition(torch.ops.aten.var)
@out_wrapper()
def var(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
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


@out_wrapper()
def std(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
) -> TensorLikeType:
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    correction = utils.set_correction(unbiased, correction)
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None

    opmath_dtype, dtype = utils.reduction_dtypes(
        a, REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    )

    result = _reduction(
        a,
        partial(prims.var, correction=correction),
        dims=dim,
        keepdims=keepdim,
        dtype=opmath_dtype,
        out=None,
        has_identity=True,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    )
    result = sqrt(result)
    return _maybe_convert_to_dtype(result, dtype)  # type: ignore[return-value,arg-type]


@register_decomposition(torch.ops.aten.mean)
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
    if dtype is None:
        dtype = a.dtype
    # can't use out wrapper because of this argument
    if out is not None and out.dtype != dtype:
        raise RuntimeError("expected out dtype and dtype to match")
    result = _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=None,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE,
    )
    if utils.is_integer_dtype(dtype):
        raise RuntimeError("result type should be floating point or complex")
    if isinstance(dim, Dim):
        dim = (dim,)  # type: ignore[assignment]
    dims = utils.reduction_dims(a.shape, dim)  # type: ignore[arg-type]
    nelem = 1 if a.ndim == 0 else reduce(operator.mul, (a.shape[i] for i in dims), 1)
    result = true_divide(result, nelem)
    result_dtype = a.dtype if dtype is None else dtype
    result = _maybe_convert_to_dtype(result, result_dtype)  # type: ignore[assignment]
    if out is not None:
        assert isinstance(out, TensorLike)
        out = _maybe_resize_out(out, result.shape)
        return _safe_copy_out(copy_from=result, copy_to=out)  # type: ignore[arg-type]
    return result


@register_decomposition(torch.ops.aten.std_mean.correction)
def std_mean(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    *,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    correction: Optional[int] = None,
):
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    s = std(a, dim, unbiased, keepdim, correction=correction)
    m = mean(a, dim, keepdim)
    return s, m


@register_decomposition(torch.ops.aten.var_mean)
def var_mean(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
):
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    v = var(a, dim, unbiased, keepdim, correction=correction)
    m = mean(a, dim, keepdim)
    return v, m


@register_decomposition(torch.ops.aten.addr)
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
    check(
        vec1.ndim == 1,
        lambda: f"addr: Expected 1-D argument vec1, but got {vec1.ndim}-D",
    )
    check(
        vec2.ndim == 1,
        lambda: f"addr: Expected 1-D argument vec2, but got {vec2.ndim}-D",
    )
    self = self.expand(vec1.shape[0], vec2.shape[0])
    if utils.is_boolean_dtype(self.dtype):
        # Integers are accepted for booleans
        check(
            is_weakly_lesser_type(type(beta), int),
            lambda: f"expected bool/int beta but got {type(beta)}",
        )
        check(
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
        check(
            is_weakly_lesser_type(type(beta), dtype_to_type(self.dtype)),
            lambda: f"cannot safely convert {type(beta)} to {self.dtype}",
        )
        check(
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
    a: TensorLikeType, size: ShapeType, stride: StrideType, storage_offset: int = 0
) -> TensorLikeType:
    return prims.as_strided(a, size, stride, storage_offset)


def broadcast_shapes(*shapes) -> ShapeType:
    return torch.Size(_broadcast_shapes(*shapes))


@torch.ops.aten.broadcast_tensors.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@torch.ops.aten.broadcast_tensors.default.py_impl(DispatchKey.Meta)
def broadcast_tensors(*tensors) -> List[TensorLikeType]:
    if len(tensors) == 1 and not isinstance(tensors[0], Tensor):
        tensors = tensors[0]
    return list(_maybe_broadcast(*tensors, preserve_cpu_scalar_tensors=False))


# CompositeImplicitAutograd - don't register decomp
def broadcast_to(a: TensorLikeType, size: ShapeType) -> TensorLikeType:
    start = len(size) - len(a.shape)
    dims = tuple(range(start, len(a.shape) + start))
    return prims.broadcast_in_dim(a, size, dims)


@register_decomposition(torch.ops.aten.cat)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("tensors",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def cat(tensors: TensorSequenceType, dim: int = 0) -> TensorLikeType:
    if len(tensors) == 0:
        msg = "cat expects at least one tensor, but received zero!"
        raise ValueError(msg)

    for tensor in tensors:
        assert isinstance(tensor, TensorLike)

    utils.check_same_device(*tensors, allow_cpu_scalar_tensors=False)

    for t in tensors:
        # match logic in legacy_cat_wrap_dim
        if t.ndim == 1 and t.size(0) == 0:
            continue
        dim = utils.canonicalize_dim(t.ndim, dim)
        utils.validate_idx(t.ndim, dim)
        break

    # Filters tensors with one dimension of length zero
    filtered = tuple(x for x in tensors if not (x.ndim == 1 and x.numel() == 0))
    if len(filtered) == 0:
        t = tensors[0]

        # TODO: fix this to work with meta tensors
        try:
            requires_grad = any(x.requires_grad for x in tensors)
        except Exception:
            requires_grad = False

        return empty((0,), dtype=t.dtype, device=t.device, requires_grad=requires_grad)

    return prims.cat(filtered, dim)


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
@register_decomposition(torch.ops.aten.constant_pad_nd)
def constant_pad_nd(
    input: TensorLikeType, pad: List[int], value: NumberType = 0
) -> TensorLikeType:
    check(
        len(pad) % 2 == 0,
        lambda: f"Length of pad must be even but instead it equals {len(pad)}",
    )

    input_sizes = input.shape
    l_inp = len(input_sizes)

    l_pad = len(pad) // 2
    l_diff = l_inp - l_pad

    check(
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
        check(
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
    check(
        memory_format != torch.preserve_format,
        lambda: "preserve memory format is unsupported by the contiguous operator",
    )

    if utils.is_contiguous_for_memory_format(a, memory_format=memory_format):
        return a

    return torch.clone(a, memory_format=memory_format)


@out_wrapper()
def dstack(tensors: TensorSequenceType) -> TensorLikeType:
    check(len(tensors) > 0, lambda: "dstack expects a non-empty TensorList")
    aligned_tensors = atleast_3d(*tensors)
    return cat(aligned_tensors, 2)


@register_decomposition(torch.ops.aten.expand)
def expand(a: Tensor, *shape) -> Tensor:
    # NOTE: cannot use utils.extract_shape_from_varargs here
    # because that also validates the shape, but the shape
    # given to expand may be "invalid"
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = tuple(shape[0])

    check(
        len(shape) >= len(a.shape),
        lambda: "expand: the requested shape has too few dimensions!",
    )

    offset = len(shape) - len(a.shape)
    shape_ = list(shape)
    for idx, x in enumerate(a.shape):
        offset_idx = idx + offset
        requested_length = shape[offset_idx]
        check(
            requested_length == x or x == 1 or requested_length == -1,
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
        msg = "Expected at least one chunk, but got {0}!".format(chunks)
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


# Note: flatten, unlike prim.collapse and prim.collapse_view has an inclusive end_dim
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
    new_shape, new_strides = prims._collapse_view_helper(a, start_dim, end_dim + 1)
    if new_shape is not None:
        return prims.collapse_view(a, start_dim, end_dim + 1)

    # Makes a copy if it can't make a view
    return prims.collapse(a, start_dim, end_dim + 1)


@register_decomposition(torch.ops.aten.flip)
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
def narrow(a: TensorLikeType, dim: int, start: int, length: int) -> TensorLikeType:
    dim = utils.canonicalize_dim(a.ndim, dim)
    return prims.slice_in_dim(a, start, start + length, axis=dim)


@register_decomposition(torch.ops.aten.narrow_copy)
@out_wrapper()
def narrow_copy(a: TensorLikeType, dim: int, start: int, length: int) -> TensorLikeType:
    # TODO: This must return a sparse tensor if the input is sparse, but refs
    # have no sparse support.  See narrow_copy_sparse in core.
    if a.is_sparse:
        raise NotImplementedError("narrow_copy ref doesn't support sparse tensors")
    return torch.clone(torch.narrow(a=a, dim=dim, start=start, length=length))  # type: ignore[call-overload]


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


@register_decomposition(torch.ops.aten.native_group_norm.default)
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
    utils.check(
        input.ndim >= 2,
        lambda: f"Expected at least 2 dimensions for input tensor but received {input.ndim}",
    )
    utils.check(
        num_channels % num_groups == 0,
        lambda: "Expected number of channels in input to be divisible by num_groups, "
        + f"but got input of shape {input.shape} and num_groups = {num_groups}",
    )

    input_reshaped = torch.reshape(
        input,
        [batch_size, num_groups, num_channels // num_groups, flattened_inner_size],
    )
    reduction_dims = utils.canonicalize_dims(input_reshaped.ndim, [-2, -1])
    out, mean, rstd = _normalize(input_reshaped, reduction_dims, eps)
    out = out.view(input.shape)

    broadcast_dims = [0] + list(dim for dim in range(2, input.ndim))
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

    mean = prims._squeeze_aten(mean, reduction_dims)
    rstd = prims._squeeze_aten(rstd, reduction_dims)
    return (out, mean, rstd)


@register_decomposition(torch.ops.aten.native_layer_norm)
def native_layer_norm(
    input: Tensor,
    normalized_shape: ShapeType,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    normalized_ndim = len(normalized_shape)
    utils.check(
        normalized_ndim >= 1,
        lambda: "Expected normalized_shape to be at least 1-dimensional, i.e., "
        + "containing at least one element, but got normalized_shape = "
        + str(normalized_shape),
    )
    # torch.Size([1, 2, 3]) == [1, 2, 3] evaluates to False
    # while torch.Size([1, 2, 3]) == (1, 2, 3) is True
    # therefore we use tuple(normalized_shape)
    utils.check(
        weight is None or weight.shape == tuple(normalized_shape),
        lambda: "Expected weight to be of same shape as normalized_shape, but got "
        + "weight of shape "
        + str(weight.shape)  # type: ignore[union-attr]
        + " and normalized_shape = "
        + str(normalized_shape),
    )
    utils.check(
        bias is None or bias.shape == tuple(normalized_shape),
        lambda: "Expected bias to be of same shape as normalized_shape, but got "
        + "bias of shape "
        + str(bias.shape)  # type: ignore[union-attr]
        + " and normalized_shape = "
        + str(normalized_shape),
    )
    utils.check(
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
@register_decomposition(torch.ops.aten.permute)
def permute(a: TensorLikeType, *dims) -> TensorLikeType:
    _permutation = utils.canonicalize_dims(
        a.ndim, utils.extract_dims_from_varargs(dims)
    )
    return prims.transpose(a, _permutation)


# Get the new shape and stride after applying unfold to an input tensor
def _get_unfold_shape_stride(
    a_shape: ShapeType, a_stride: StrideType, dimension: int, size: int, step: int
):
    a_ndim = len(a_shape)
    dim = utils.canonicalize_dim(a_ndim, dimension, wrap_scalar=True)
    max_size = 1 if a_ndim == 0 else a_shape[dim]
    last_stride = 1 if a_ndim == 0 else a_stride[dim]

    utils.check(
        size <= max_size,
        lambda: f"Maximum size for tensor at dimension {dim} is {max_size} but size is {size}",
    )

    utils.check(
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


@register_decomposition(torch.ops.aten.repeat)
def repeat(a: Tensor, *repeat_shape) -> Tensor:
    repeat_shape = utils.extract_shape_from_varargs(repeat_shape, validate=False)
    utils.check(
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
    enumerated_stride.sort(key=lambda item: item[1], reverse=True)
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
    # Creates a valid shape
    shape = utils.extract_shape_from_varargs(shape, validate=False)
    # Reshape may be given a shape with a -1 length
    # This indicates that the dimension's length should be inferred
    shape = utils.infer_size(shape, a.numel())

    # Short-circuits if shape is the same
    if tuple(a.shape) == tuple(shape):
        return prims.view_of(a)

    # Special-cases tensors with no elements
    if a.numel() == 0:
        return as_strided(a, shape, utils.make_contiguous_strides_for(shape))

    # Special-cases reshaping zero dim tensors
    if a.ndim == 0:
        _a = a
        for length in shape:
            assert length == 1
            _a = unsqueeze(_a, -1)
        return _a

    # Special-cases reshaping to zero dim tensors
    if len(shape) == 0:
        _a = a
        for length in a.shape:
            assert length == 1
            _a = squeeze(_a, -1)
        return _a

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
        if length == a_.shape[idx]:
            idx = idx + 1
            continue

        # Gathers enough original dimensions such that this new dimension can be created
        # Note that this accumulation will terminate because we've verified a and the shape
        # specify the same number of elements above
        accum = a_.shape[idx]
        end = idx
        while accum % length != 0:
            end = end + 1
            accum = accum * a_.shape[end]
        if end != idx:
            # NOTE: in this case multiple dimensions must be flatten to create the desired dimension
            # This flattening is why reshape sometimes creates a copy -- because flattening
            # may return a view of a copy

            # Checks if collapse can be a view and short-circuits to copying reshape if it can't
            new_shape, new_strides = prims._collapse_view_helper(a_, idx, end + 1)
            if new_shape is None:
                if allow_copy:
                    return prims.reshape(a, shape)

                msg = "Cannot view a tensor with shape {0} and strides {1} as a tensor with shape {2}!".format(
                    a.shape, a.stride(), shape
                )
                raise ValueError(msg)

            a_ = flatten(a_, idx, end)

        # Splits the (possibly flattened) dimension to create the desired dim length
        if accum != length:
            a_ = prims.split_dim(a_, idx, length)

        idx = idx + 1

    # Squeezes tail
    while idx < a_.ndim:
        assert a_.shape[idx] == 1
        a_ = squeeze(a_, idx)

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


@register_decomposition(torch.ops.aten.roll)
def roll(
    a: TensorLikeType, shifts: DimsType, dims: DimsType = tuple()
) -> TensorLikeType:
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
        return clone(a)

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
        first_dim_rolled = torch.roll(a, shifts[0], dims[0])
        return torch.roll(first_dim_rolled, tail_shifts, tail_dims)

    # This path is taken when only one dimension is rolled
    # For example to get `first_dim_rolled` above
    dim = dims[0]
    size = a.shape[dim]
    start = (size - shifts[0]) % size
    t0 = torch.narrow(a, dim, start, size - start)
    t1 = torch.narrow(a, dim, 0, start)
    return torch.cat((t0, t1), dim)


@register_decomposition(torch.ops.aten.rot90)
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
        return clone(a, memory_format=torch.contiguous_format)


def _check_stack_inputs(tensors: TensorSequenceType) -> None:
    entry_shape = tensors[0].shape
    for i in range(1, len(tensors)):
        assert tensors[i].shape == entry_shape, (
            f"stack expects each tensor to be equal size, but got {entry_shape} at entry 0"
            f"and {tensors[i].shape} at entry {i}"
        )


@register_decomposition(torch.ops.aten.stack)
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
    a_max = amax(a_, dim, keepdim=True)
    a_exp = exp(a_ - a_max)
    return _maybe_convert_to_dtype(
        true_divide(a_exp, sum(a_exp, dim, keepdim=True)), result_dtype
    )  # type: ignore[return-value]


# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
def hstack(tensors: TensorSequenceType) -> TensorLikeType:
    check(len(tensors) > 0, lambda: "hstack expects a non-empty TensorList")
    aligned_tensors = atleast_1d(*tensors)
    if aligned_tensors[0].ndim == 1:
        return cat(aligned_tensors, 0)
    return cat(aligned_tensors, 1)


# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
def vstack(tensors: TensorSequenceType) -> TensorLikeType:
    check(len(tensors) > 0, lambda: "vstack expects a non-empty TensorList")
    aligned_tensors = atleast_2d(*tensors)
    return cat(aligned_tensors, 0)


# CompositeImplicitAutograd - don't register decomp
def unflatten(a: TensorLikeType, dim: int, sizes: ShapeType) -> TensorLikeType:
    dim = utils.canonicalize_dim(a.ndim, dim)
    utils.check(len(sizes) != 0, lambda: "unflatten: sizes must be non-empty")
    return a.view(tuple(a.shape[:dim]) + tuple(sizes) + tuple(a.shape[dim + 1 :]))


@register_decomposition(torch.ops.aten.unbind)
def unbind(t: TensorLikeType, dim: int = 0) -> TensorSequenceType:
    dim = utils.canonicalize_dim(t.ndim, dim)
    check(
        len(t.shape) > 0,
        lambda: "dimension specified as 0 but tensor has no dimensions",
        IndexError,
    )
    return tuple(
        torch.squeeze(s, dim) for s in torch.tensor_split(t, t.shape[dim], dim)
    )


@register_decomposition(torch.ops.aten.index_copy)
@out_wrapper()
def index_copy(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike):
    return x.clone(memory_format=torch.contiguous_format).index_copy_(
        dim, index, tensor
    )


@register_decomposition(torch.ops.aten.index_copy_)
def index_copy_(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike):
    dim = utils.canonicalize_dims(x.ndim, dim)
    utils.check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    # Treat scalars as elements of \R^1
    y = x.unsqueeze(0) if x.ndim == 0 else x
    idx = (slice(None),) * dim + (index,)
    y[idx] = tensor
    return x


@register_decomposition(torch.ops.aten.index_fill)
def index_fill(
    x: TensorLike, dim: int, index: TensorLike, value: Union[NumberType, TensorLike]
):
    return x.clone().index_fill_(dim, index, value)  # type: ignore[arg-type]


@register_decomposition(torch.ops.aten.index_fill_)
def index_fill_(
    x: TensorLike, dim: int, index: TensorLike, value: Union[NumberType, TensorLike]
):
    if isinstance(value, TensorLike):
        utils.check(
            value.ndim == 0,
            lambda: "Only supports 0-dimensional value tensor. "  # type: ignore[union-attr]
            f"Got a tensor with {value.ndim} dimensions.",
        )  # type: ignore[arg-type]
        return x.clone().index_copy_(dim, index, value)
    dim = utils.canonicalize_dims(x.ndim, dim)
    utils.check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    idx = (slice(None),) * dim + (index,)
    # Treat scalars as elements of \R^1
    y = x.unsqueeze(0) if x.ndim == 0 else x
    y[idx] = value  # type: ignore[assignment]
    return x


@register_decomposition(torch.ops.aten.index_add)
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


@register_decomposition(torch.ops.aten.index_select)
@out_wrapper()
def index_select(x: TensorLike, dim: int, index: TensorLike):
    dim = utils.canonicalize_dims(x.ndim, dim)
    utils.check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    # Treat scalars as elements of \R^1
    if x.ndim == 0:
        # we cannot write `x.unsqueeze(0)[index].squeeze(0).clone()`
        # as tensor[index] will trigger index.item() if index is a 0-dim tensor
        # and .item() cannot be symbolically traced with FakeTensor.
        return torch.ops.aten.index(x.unsqueeze(0), [index]).squeeze(0).clone()
    idx = (slice(None),) * dim + (index,)
    return x[idx]


# Note: although squeeze is documented as having the out= kwarg it doesn't
@register_decomposition(torch.ops.aten.squeeze)
def squeeze(a: TensorLikeType, dim: Optional[int] = None) -> TensorLikeType:
    if dim is not None:
        dim = utils.canonicalize_dim(a.ndim, dim)
        # Short-circuits if the tensor has no dimensions
        if len(a.shape) == 0:
            assert dim == 0
            return prims.view_of(a)

        # Note: squeeze does not modify tensors when the given dim is not a dimension of length 1
        if a.shape[dim] != 1:
            return prims.view_of(a)
        return prims.squeeze(a, (dim,))

    dims = tuple(idx for idx in range(len(a.shape)) if a.shape[idx] == 1)
    return prims.squeeze(a, dims)


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
            msg = "tensor_split: if indices_or_sections is a tensor it must be on the CPU, but received one on {0}".format(
                indices_or_sections.device
            )
            raise ValueError(msg)
        if indices_or_sections.dtype != torch.long:
            msg = "tensor_split: if indices_or_sections is a tensor it must have long dtype, "
            " but received one with dtype {0}".format(indices_or_sections.dtype)
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
            msg = "tensor_split: number of sections must be greater than 0, but was {0}".format(
                sections
            )
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
                "but received a tensor with {0} dimensions".format(
                    indices_or_sections.ndim
                )
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
    check(
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
        check(
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

    check(
        isinstance(indices_or_sections, (list, tuple)),
        lambda: (
            "hsplit(): received an invalid combination of arguments. "
            "Expected indices_or_sections to be of type int, list of ints or tuple of ints "
            f"but got type {type(indices_or_sections)}"
        ),
        exc_type=TypeError,
    )

    split_sizes = indices_or_sections
    return tensor_split(a, split_sizes, dim)


# CompositeImplicitAutograd - don't register decomp
def vsplit(
    a: TensorLikeType, indices_or_sections: DimsType
) -> Tuple[TensorLikeType, ...]:
    check(
        a.ndim >= 2,
        lambda: (
            "torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with "
            + str(a.ndim)
            + " dimensions!"
        ),
    )
    if isinstance(indices_or_sections, IntLike):
        split_size = indices_or_sections
        check(
            (split_size != 0 and a.shape[0] % split_size == 0),
            lambda: (
                "torch.vsplit attempted to split along dimension 0 "
                + ", but the size of the dimension "
                + str(a.shape[0])
                + " is not divisible by the split_size "
                + str(split_size)
                + "!"
            ),
        )
        return tensor_split(a, split_size, 0)

    check(
        isinstance(indices_or_sections, (list, tuple)),
        lambda: (
            "vsplit(): received an invalid combination of arguments. "
            "Expected indices_or_sections to be of type int, list of ints or tuple of ints "
            f"but got type {type(indices_or_sections)}"
        ),
        exc_type=TypeError,
    )

    split_sizes = indices_or_sections
    return tensor_split(a, split_sizes, 0)


@register_decomposition(torch.ops.aten.diag.out)
@out_wrapper()
def diag(
    self: TensorLikeType,
    offset: int = 0,
) -> TensorLikeType:
    ndim = self.dim()
    utils.check(
        ndim in (1, 2), lambda: f"diag(): Supports 1D or 2D tensors. Got {ndim}D"
    )
    if ndim == 1:
        return torch.diag_embed(self, offset)
    else:
        return torch.diagonal_copy(self, offset)


@register_decomposition(torch.ops.aten.diagonal_scatter)
@out_wrapper()
def diagonal_scatter(
    input: TensorLikeType,
    src: TensorLikeType,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
) -> TensorLikeType:
    out = input.clone()
    diag = out.diagonal(offset, dim1, dim2)
    check(
        diag.shape == src.shape,
        lambda: "expected src to have a size equal to the diagonal of the input."
        f"Got {src.shape} for a diagonal of shape {diag.shape}",
    )
    copy_to(diag, src)
    return out


@register_decomposition(torch.ops.aten.diagonal)
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

    check(
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


diagonal_copy = _make_copy_from_view(diagonal)


@register_decomposition(torch.ops.aten.diag_embed)
def diag_embed(
    t: TensorLikeType,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> TensorLikeType:
    """
    Reference implementation of torch.diag_embed
    """
    # as per the docs, exchanging dims is equivalent to changing the sign of
    # offset
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
        offset = -offset

    # convert from negative dims
    rank = t.ndim + 1
    dim1 = utils.canonicalize_dim(rank=rank, idx=dim1)
    dim2 = utils.canonicalize_dim(rank=rank, idx=dim2)

    check(
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


# CompositeImplicitAutograd - don't register decomp
def dsplit(a: TensorLikeType, sections: DimsType) -> TensorSequenceType:
    if a.ndim < 3:
        raise RuntimeError(
            f"torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with {a.ndim} dimensions!"
        )
    if isinstance(sections, IntLike) and (sections == 0 or a.shape[2] % sections != 0):
        raise RuntimeError(
            "torch._refs.dsplit attempted to split along dimension 2, "
            + f"but the size of the dimension {a.shape[2]} is not divisible by the split_size {sections}!"
        )
    return tensor_split(a, sections, 2)


@register_decomposition(torch.ops.aten.t.default)
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
    check(
        a.ndim in (0, 2),
        lambda: (
            "The use of `x.T` on tensors of dimension other than 0 or 2 "
            "to reverse their shape is not supported."
        ),
    )
    return a.t()


@register_decomposition(torch.ops.aten.transpose)
def transpose(a: TensorLikeType, dim0: int, dim1: int) -> TensorLikeType:
    _dim0, _dim1 = utils.canonicalize_dims(a.ndim, (dim0, dim1))  # type: ignore[misc]

    if a.ndim <= 1 or dim0 == dim1:
        return prims.view_of(a)

    _permutation = list(range(0, a.ndim))
    _permutation[_dim0] = _dim1
    _permutation[_dim1] = _dim0
    return torch.permute(a, _permutation)


# Aliases for transpose
swap_axes = transpose


@register_decomposition(torch.ops.aten.unfold)
def unfold(
    self: TensorLikeType, dimension: int, size: int, step: int
) -> TensorLikeType:
    shape, strides = _get_unfold_shape_stride(
        self.shape, self.stride(), dimension, size, step
    )
    return self.as_strided(shape, strides)


@register_decomposition(torch.ops.aten.unfold_copy)
@out_wrapper()
def unfold_copy(self: TensorLikeType, dimension: int, size: int, step: int):
    return self.unfold(dimension, size, step).clone(
        memory_format=torch.contiguous_format
    )


@register_decomposition(torch.ops.aten.cumsum)
def cumsum(
    a: TensorLikeType,
    dim: int,
    *,
    keepdim: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # We implement all the kwargs of a reduction. ATen just handles dtype
    # nb. This decomposition may not be as efficient as a backend-specific implementation
    ndim = a.ndim
    dim = utils.canonicalize_dim(ndim, dim)
    if ndim == 0:
        return sum(a.unsqueeze(0), dim=0, keepdim=keepdim, dtype=dtype, out=out)
    a = a.unsqueeze(dim + 1)
    rg = torch.arange(a.shape[dim], device=a.device)
    mask = rg.unsqueeze(1) <= rg
    for _ in range(ndim - dim - 1):
        mask = mask.unsqueeze(-1)
    masked_a = utils.mask_tensor(mask, a)
    return sum(masked_a, dim=dim, keepdim=keepdim, dtype=dtype, out=out)


@register_decomposition(torch.ops.aten.unsqueeze)
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
@register_decomposition(torch.ops.aten.view)
def view(a: TensorLikeType, *shape: ShapeType) -> TensorLikeType:
    return _reshape_view_helper(a, *shape, allow_copy=False)


# CompositeImplicitAutograd - don't register decomp
def view_as(self: TensorLikeType, other: TensorLikeType) -> TensorLikeType:
    return self.view(other.size())


# CompositeImplicitAutograd - don't register decomp
def ravel(a: TensorLikeType) -> TensorLikeType:
    return reshape(a, (-1,))


@register_decomposition(torch.ops.aten.empty.memory_format)
@out_wrapper()
def empty(
    *shape,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> TensorLikeType:
    check(
        memory_format != torch.preserve_format,
        lambda: "torch.empty: the Preserve memory format is not supported",
    )

    shape = utils.extract_shape_from_varargs(shape)

    if memory_format == torch.contiguous_format:
        strides = utils.make_contiguous_strides_for(shape)
    elif memory_format == torch.channels_last_3d:
        strides = utils.make_channels_last_3d_strides_for(shape)
    else:  # memory_format == torch.channels_last
        check(
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


@register_decomposition(torch.ops.aten.new_empty)
def new_empty(
    a: TensorLikeType,
    size: ShapeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
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


@register_decomposition(torch.ops.aten.new_empty_strided)
def new_empty_strided(
    a: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
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


@register_decomposition(torch.ops.aten.zeros.default)
@out_wrapper()
def zeros(
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
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


@register_decomposition(torch.ops.aten.new_zeros)
def new_zeros(
    a: TensorLikeType,
    size: ShapeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    return torch.full(
        size,
        False if dtype == torch.bool else 0,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_decomposition(torch.ops.aten.ones.default)
@out_wrapper()
def ones(
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
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


@register_decomposition(torch.ops.aten.new_ones)
def new_ones(
    a: TensorLikeType,
    size: ShapeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    return torch.full(
        size,
        True if dtype == torch.bool else 1,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_decomposition(torch.ops.aten.new_full)
def new_full(
    a: TensorLikeType,
    size: ShapeType,
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
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


@register_decomposition(torch.ops.aten.empty_like)
def empty_like(
    a: TensorLikeType,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    layout: Optional[torch.layout] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> TensorLikeType:

    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    strides: Tuple[int, ...]

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
    strides = utils.compute_elementwise_output_strides(a)
    return torch.empty_strided(
        a.shape,
        strides,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_decomposition(
    [
        torch.ops.aten.arange.default,
        torch.ops.aten.arange.start,
        torch.ops.aten.arange.start_step,
    ]
)
@out_wrapper()
def arange(
    start: NumberType = 0,
    end: Optional[NumberType] = None,
    step: NumberType = 1,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    utils.check_layout(layout)
    utils.check_pin_memory(pin_memory)
    # Case: torch.arange(5)
    if end is None:
        end = start
        start = 0
    return prims.arange(
        start,
        end,
        step,
        dtype=dtype,
        # layout=layout,
        device=device,
        # pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_decomposition(torch.ops.aten.lerp)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("start", "end", "weight"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def lerp(start: Tensor, end: Tensor, weight: Union[Tensor, NumberType]):
    check(
        start.dtype == end.dtype,
        lambda: f"expected dtype {start.dtype} for `end` but got dtype {end.dtype}",
    )
    if isinstance(weight, Number):
        weight = start.new_full((), weight)  # type: ignore[arg-type]
    else:
        check(
            start.dtype == weight.dtype,
            lambda: f"expected dtype {start.dtype} for `weight` but got dtype {weight.dtype}",  # type: ignore[union-attr]
        )
    assert isinstance(weight, Tensor)  # mypy
    # We implement it this way for numerical stability. We assume (in the stability optimisation)
    # that 0 <= weight <= 1. We take the abs to deal with comples numbers
    # We want to do operations near zero, which is where floating points are most precise
    # If weight.abs() >= 0.5:
    #    return (1 - weight) * (start - end) + end
    mask = weight.abs() >= 0.5
    coeff = torch.where(mask, weight - 1, weight)
    base = torch.where(mask, end, start)
    return coeff * (end - start) + base


@register_decomposition(torch.ops.aten.linspace)
@out_wrapper()
def linspace(
    start: NumberType,
    end: NumberType,
    steps: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    layout: torch.layout = torch.strided,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    if dtype is None:
        dtype = torch.get_default_dtype()

    # NB: NumPy actually doesn't do this cast, but for this ref, I'd rather have this
    #     cast than not, because it allows us to always go into the precise path
    #     if dtype is integral and not worry about whether start/end are float
    if prims.utils.is_integer_dtype(dtype):
        if isinstance(start, FloatLike):
            start = sym_int(start)
        if isinstance(end, FloatLike):
            end = sym_int(end)

    if py_any(isinstance(arg, complex) for arg in (start, end, steps)):
        raise NotImplementedError
    assert not isinstance(start, complex) and not isinstance(end, complex)  # for mypy

    check(
        isinstance(steps, IntLike),
        lambda: "steps must be int, not float",
        exc_type=TypeError,
    )
    assert isinstance(steps, IntLike)  # for mypy
    check(steps >= 0, lambda: "number of steps must be non-negative")

    factory_kwargs = {
        "layout": layout,
        "device": device,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
    }
    if steps == 0:
        ret = torch.full((0,), 0, dtype=dtype, **factory_kwargs)  # type: ignore[call-overload]
    elif steps == 1:
        ret = torch.full((1,), start, dtype=dtype, **factory_kwargs)  # type: ignore[call-overload]
    elif start == end:
        ret = torch.full((steps,), start, dtype=dtype, **factory_kwargs)  # type: ignore[call-overload]
    else:
        if prims.utils.is_integer_dtype(dtype):
            # We need to cast to int, so to avoid off-by-one issues
            # do the entire computation with ints when we can
            assert isinstance(start, IntLike) and isinstance(end, IntLike)
            step_size_x_denom = end - start
            eps = 1 if end > start else -1
            denom = steps - 1
            ret = prims.to_dtype(
                torch.arange(
                    start * denom,
                    end * denom + eps,
                    step_size_x_denom,
                    dtype=torch.int64,
                    **factory_kwargs,  # type: ignore[arg-type]
                )
                / denom,
                dtype,
            )
        else:
            step_size = (end - start) / (steps - 1)
            eps = step_size / 2
            ret = prims.to_dtype(
                torch.arange(  # type: ignore[call-overload]
                    start, end + eps, step_size, dtype=torch.float64, **factory_kwargs
                ),
                dtype,
            )

    return ret


@register_decomposition(torch.ops.aten.logspace)
@out_wrapper()
def logspace(
    start: NumberType,
    end: NumberType,
    steps: NumberType,
    base: NumberType = 10,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
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
        if isinstance(end, FloatLike):
            end = sym_int(end)

    assert not isinstance(base, complex)  # for mypy
    if base < 0:
        raise NotImplementedError
    ret = torch.linspace(
        start,
        end,
        steps,
        dtype=torch.float64,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )
    return prims.to_dtype(torch.pow(base, ret), dtype)


@overload
def meshgrid(tensors: Sequence[TensorLikeType], indexing: str):
    pass


@overload
def meshgrid(*tensors: TensorLikeType, indexing: str):
    pass


@register_decomposition(torch.ops.aten.meshgrid)
def meshgrid(
    *tensors: Union[TensorLikeType, List[TensorLikeType], Tuple[TensorLikeType]],
    indexing: str,
) -> List[TensorLikeType]:
    # This ref simultaneously handles two overloads (see stubs above)
    # The `indexing` argument is currently optional for torch.meshgrid, but we
    # plan to make the argument required: https://github.com/pytorch/pytorch/issues/50276
    if isinstance(tensors[0], list) or isinstance(tensors[0], tuple):
        assert len(tensors) == 1
        tensors = tuple(tensors[0])

    check(
        py_all(isinstance(a, TensorLike) for a in tensors),
        lambda: "meshgrid expects its inputs to be tensors",
    )

    check(len(tensors) > 0, lambda: "meshgrid expects a non-empty TensorList")

    for i in range(len(tensors) - 1):
        check(
            tensors[i].dtype == tensors[i + 1].dtype,  # type: ignore[union-attr]
            lambda: "meshgrid expects all tensors to have the same dtype",
        )
        check(
            tensors[i].device == tensors[i + 1].device,  # type: ignore[union-attr]
            lambda: "meshgrid expects all tensors to have the same device",
        )

    swap_first_and_second_tensors = False
    if indexing == "xy":
        swap_first_and_second_tensors = len(tensors) >= 2
        if swap_first_and_second_tensors:
            tensors = (tensors[1], tensors[0], *tensors[2:])
    else:
        check(
            indexing == "ij",
            lambda: (
                'torch.meshgrid: indexing must be one of "xy" or "ij", '
                f"but received: {indexing}"
            ),
        )

    result_shape: List[int] = []
    for t in tensors:
        assert isinstance(t, TensorLike)  # mypy
        check(
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

    utils.check(
        len(source) == len(destination),  # type: ignore[arg-type]
        lambda: (
            "movedim: Invalid source or destination dims: source "
            f"({source} dims) should contain the same number of dims as "
            f"destination ({destination} dims)"
        ),
    )

    rank = input.ndim
    ss = tuple(utils.canonicalize_dims(rank=rank, indices=source))  # type: ignore[arg-type]
    ds = tuple(utils.canonicalize_dims(rank=rank, indices=destination))  # type: ignore[arg-type]

    sss = set(ss)
    dss = set(ds)

    utils.check(
        len(ss) == len(sss),
        lambda: f"movedim: repeated dim in `source` {source}",
    )
    utils.check(
        len(ds) == len(dss),
        lambda: f"movedim: repeated dim in `destination` {destination}",
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
@register_decomposition(torch.ops.aten.empty_strided)
def empty_strided(
    shape: Union[ShapeType, Tuple[ShapeType]],
    strides: StrideType,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
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


@register_decomposition(torch.ops.aten.eye)
@out_wrapper()
def eye(
    n: int,
    m: Optional[int] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,  # TODO: unused
) -> TensorLikeType:
    """
    Reference implementation of torch.eye
    """
    if m is None:
        m = n

    check(n >= 0, lambda: f"n must be greater or equal to 0, got {n}")
    check(m >= 0, lambda: f"m must be greater or equal to 0, got {m}")

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


@out_wrapper()
def full(
    shape: ShapeType,
    fill_value: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    e = empty(
        shape,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )
    return fill(e, fill_value)


def full_like(
    a: TensorLikeType,
    fill_value: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
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


zeros_like = partial(full_like, fill_value=False)


ones_like = partial(full_like, fill_value=True)

# TODO: add pin_memory support
@register_decomposition(torch.ops.aten.randn.default)
@out_wrapper()
def randn(
    *shape,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    layout: Optional[torch.layout] = None,
    requires_grad: bool = False,
    pin_memory: Optional[bool] = None,
) -> TensorLikeType:

    check(pin_memory is None, lambda: "pin_memory parameter is not supported!")

    shape_ = utils.extract_shape_from_varargs(shape)

    dtype = utils.dtype_or_default(dtype)
    device = utils.device_or_default(device)
    layout = utils.layout_or_default(layout)

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
    device: Optional[torch.device] = None,
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


@register_decomposition(torch.ops.aten.uniform)
def uniform(
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

    return prims.uniform(shape, low=low, high=high, dtype=dtype, device=device)


@register_decomposition(
    [torch.ops.aten.masked_fill.Scalar, torch.ops.aten.masked_fill.Tensor]
)
def masked_fill(a: TensorLikeType, mask: TensorLikeType, value: TensorOrNumberLikeType):
    python_type = utils.dtype_to_type(a.dtype)
    if isinstance(value, Number):
        value_type = type(value)
    else:
        # NOTE: Could not use value = item(value) as it resulted in
        # RuntimeError: Cannot cast FakeTensor(cpu) to number
        value_ndim = value.ndim
        check(
            value_ndim == 0,
            lambda: f"only supports a 0-dimensional value tensor, but got tensor with {value_ndim} dimension",
        )
        # `masked_fill` allows cpu scalar to be moved to cuda but not otherwise.
        check(
            a.device.type == "cuda" or value.device == a.device,
            lambda: "Expected `value` to be on same device as `a`",
        )
        value_type = utils.dtype_to_type(value.dtype)
        if utils.is_cpu_scalar_tensor(value):
            value = value.item()

    if value_type is complex:
        # only downcasting from complex to lower type is not allowed.
        # We allow casting `value` to lower type for other case
        # Eg. float -> int.
        # Ref: https://github.com/pytorch/pytorch/issues/79195
        check(
            utils.is_weakly_lesser_type(value_type, python_type),
            lambda: f"could not convert to type {python_type} without overflow",
        )

    # Since `where` allows type-promotion,
    # cast value to correct type before passing to `where`
    if isinstance(value, Number):
        r = torch.where(mask, python_type(value), a)
    else:
        assert isinstance(value, TensorLike)
        r = torch.where(mask, prims.to_dtype(value, a.dtype), a)

    # aten.mask_fill always return a new contiguous tensor
    # contiguous() is needed to correctly model the output stride
    return r.contiguous()


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


# TODO: add OpInfo for torch.equal and refs.equal
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


@register_decomposition(torch.ops.aten.trace)
def trace(self: TensorLikeType) -> TensorLikeType:
    utils.check(
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


@register_decomposition(torch.ops.aten.triu)
@out_wrapper()
def triu(a: TensorLikeType, diagonal: int = 0) -> TensorLikeType:
    utils.check(
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


@register_decomposition(torch.ops.aten.tril)
@out_wrapper()
def tril(a: TensorLikeType, diagonal: int = 0) -> TensorLikeType:
    utils.check(
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
    check(row >= 0, lambda: f"row must be non-negative, got {row}")
    check(col >= 0, lambda: f"col must be non-negative, got {col}")
    check(
        dtype in (torch.int32, torch.int64),
        lambda: f"\"{name}\" not implemented for '{dtype}'",
    )


# This is based on tril_indices_cuda in aten/src/ATen/native/cuda/TensorFactories.cu
@register_decomposition(torch.ops.aten.tril_indices)
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
    row_inds1 = prims.to_dtype(row_inds1 + row_offset, dtype)
    col_inds1 = prims.to_dtype(col_inds1, dtype)

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


@register_decomposition(torch.ops.aten.triu_indices)
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
    row_inds1 = prims.to_dtype(row_inds1, dtype)
    col_inds1 = prims.to_dtype(col_inds1, dtype)

    if col:
        row_inds1 = row_inds1 + (rectangle_size // col)
    col_inds1 = col_inds1 + col_offset

    return torch.stack(
        (torch.cat((row_inds2, row_inds1)), torch.cat((col_inds2, col_inds1)))
    )


@register_decomposition(torch.ops.aten.bucketize)
@out_wrapper(exact_dtype=True)
def bucketize(
    a: TensorLikeType,
    boundaries: TensorLikeType,
    *,
    out_int32: bool = False,
    right: bool = False,
):
    utils.check(
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


import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
