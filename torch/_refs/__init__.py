import torch

import torch._prims as prims
import torch._prims.utils as utils
from torch._prims.utils import (
    DimsType,
    TensorLike,
    TensorLikeType,
    DimsSequenceType,
    TensorSequenceType,
    Number,
    NumberType,
)

from functools import reduce
from enum import Enum
from typing import Sequence, Optional, Union, Callable, List, Tuple
import operator
import warnings
import math

# Experimental module containing prototype Python references for existing
#   PyTorch operations.

__all__ = [
    #
    # Elementwise Unary References
    #
    "abs",
    "acos",
    "acosh",
    "asin",
    "atan",
    # "bessel_i0e",  # special.i0e
    # "bessel_i1e",  # special.i1e
    # "cbrt",  # No corresponding torch operation
    "ceil",
    "cos",
    "cosh",
    "digamma",
    "erf",
    "erfinv",
    "erfc",
    "exp",
    "expm1",
    "floor",
    "isfinite",
    "isnan",
    "lgamma",
    "log",
    "log1p",
    "neg",
    "reciprocal",
    "round",  # TODO: model kwargs
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
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
    # "complex",
    # 'copysign', # where
    # 'div', # need to implement all rounding modes first
    "eq",
    "float_power",
    # 'floor_divide', # requires floor
    # 'fmax', # requires where
    # 'fmod',
    # 'gcd',
    "ge",
    "gt",
    # 'heaviside',
    # 'hypot',
    "igamma",
    "igammac",
    # 'isclose', # abs, sub, le, add, mul
    # 'lcm',
    # 'ldexp',
    "le",
    # 'logical_and',
    # 'logical_or',
    # 'logical_xor',
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
    # 'remainder',
    # 'rsub', # unblocked
    # # special.xlog1py
    # # special.zeta
    "sub",
    "true_divide",
    # 'xlogy', # where?, log, mul
    #
    # Conditional references
    #
    "where",  # TODO: add opinfo
    #
    # Data conversion and movement references
    #
    "copy_to",  # TODO: add opinfo
    #
    # Reduction ops
    #
    "sum",
    "amax",
    "amin",
    #
    # View & Shape Ops
    #
    "cat",
    "permute",
    "transpose",
    "swap_axes",  # alias for transpose
    "tensor_split",
]

Tensor = torch.Tensor


class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    INT_TO_FLOAT = (1,)
    ALWAYS_BOOL = (2,)
    OP_MATH = (3,)
    COMPLEX_TO_FLOAT = (4,)
    BOOL_TO_LONG = (5,)


class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME = (0,)
    SAME_OR_REAL = (1,)  # for complex types outputs corresponding real type
    OP_MATH = (2,)  # keep output in opmath type, needed for mean
    ALWAYS_BOOL = (3,)


# Maps datatypes to their computation types for elementwise operations
_computation_dtype_map = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
    torch.complex32: torch.complex64,
}


def _get_computation_dtype(dtype: torch.dtype) -> torch.dtype:
    return _computation_dtype_map.get(dtype, dtype)


# TODO: document type promotion kinds
def _elementwise_dtypes(
    *_args, type_promotion: ELEMENTWISE_TYPE_PROMOTION_KIND
) -> Tuple[torch.dtype, torch.dtype]:
    """
    Computes the computation and result dtypes for elementwise type promotion
    on the given arguments and with the given elementwise type promotion kind.

    Note that not all inputs to an elementwise operation necessarily participate in type promotion.
    For example, the "alpha" parameter of torch.add does not participate in type promotion,
    although it is cast to the Python type corresponding to the computation dtype that
    the type promotion algorithm determines.

    Default elementwise type promotion, which all other type promotion kinds tweak (see below),
    first decides which of four ordered types to use:

    bool -> integer -> floating point -> complex

    The selected type is the "lowest" type in the above list such that all number arguments
    have a weakly "lower" type and all tensor arguments have a weakly lower corresponding
    type for their dtype.

    Once the type is determined, the particular result dtype is found. The dtypes are
    partially ordered as follows:

    bool -> uint8, int8 -> int16 -> int32 -> int64 ->
      float16, bfloat16 -> float32 -> float64 -> complex32 -> complex64 -> complex128

    The result dtype is selected by:
      - if no tensor's dtype has the same corresponding type as the one selected,
          then the result dtype is the (default) dtype corresponding to the selected type
          (for example, 1.5 + an integer tensor has a result dtype of the default floating point dtype)
      - if the result type is complex then the dtype is:
        -  the default complex dtype if there are no floating point or complex tensors
        -  if there are floating point or complex tensors with one or more dimensions, then
            the complex dtype corresponding to the highest corresponding complex dtype among those tensors
            (for example, double + cfloat -> cdouble)
        -  if there are only floating point or complex tensors with zero dimensions, then
            the complex dtype corresponding to the highest corresponding complex dtype among those tensors
      - if the first two cases do not apply, the result dtype is the highest dtype among
          all tensors with one or more dimensions of the output type, and if there are no such
          tensors then it's the highest dtype among all tensors with zero dimensions of the output type
          (for example, long + half -> half, even if the half tensor has zero dimensions)

    The "corresponding complex dtypes" are:
      float16    -> complex32
      bfloat16   -> complex64
      float32    -> complex64
      float64    -> complex128
      complex32  -> complex32
      complex64  -> complex64
      complex128 -> complex128

    The DEFAULT type promotion option computes per above, and uses the result dtype as the computation dtype.

    The OP_MATH, INT_TO_FLOAT, COMPLEX_TO_FLOAT and BOOL_TO_LONG type promotion options tweak the above slightly.
    OP_MATH determines a "computation dtype" from the result dtype, and the mapping is simple:

      float16   -> float32
      bfloat16  -> float32
      complex32 -> complex64

    INT_TO_FLOAT, COMPLEX_TO_FLOAT, and BOOL_TO_LONG compute the computation type in the same way, but INT_TO_FLOAT
    and BOOL_TO_LONG map the result dtype to another dtype first, and COMPLEX_TO_FLOAT maps its result dtype
    after the compuation dtype is determined, as follows:

      INT_TO_FLOAT  maps all boolean and integer result dtypes to the default floating point dtype
      COMPLEX_TO_FLOAT  maps complex result dtypes to their corresponding floating point dtype
      BOOL_TO_LONG maps the boolean result dtype to long

    The "corresponding floating point dtypes" are:
      complex32  -> float16
      complex64  -> float32
      complex128 -> float64

    The ALWAYS_BOOL type promotion option always maps the result dtype to bool.

    Example operators for each type promotion option:
      DEFAULT          : nextafter
      OP_MATH          : add
      INT_TO_FLOAT     : sin
      COMPLEX_TO_FLOAT : abs
      BOOL_TO_LONG     : pow
      ALWAYS_BOOL      : eq

    """

    args = tuple(x for x in _args if x is not None)

    highest_type: type = bool
    for x in args:
        if not isinstance(x, (Number, TensorLike)):
            msg = (
                "Unexpected type {0} when computing elementwise type promotion!".format(
                    str(type(x))
                )
            )
            raise ValueError(msg)

        if isinstance(x, Number):
            highest_type = utils.get_higher_type(highest_type, type(x))
        else:
            # x is a TensorLike
            highest_type = utils.get_higher_type(
                highest_type, utils.dtype_to_type(x.dtype)
            )

    result_dtype = None

    def _find_highest_dtype_filtered(
        args, filter, *, float_as_complex=False, all_tensors_equal=False
    ) -> Optional[torch.dtype]:
        zero_dim_tensor_dtype = None
        one_plus_dim_tensor_dtype = None
        for x in args:
            if isinstance(x, TensorLike) and filter(x.dtype):
                _dtype = x.dtype
                if float_as_complex and utils.is_float_dtype(_dtype):
                    _dtype = utils.corresponding_complex_dtype(_dtype)
                if x.ndim == 0 and not all_tensors_equal:
                    zero_dim_tensor_dtype = utils.get_higher_dtype(
                        zero_dim_tensor_dtype, _dtype
                    )
                else:
                    # x.ndim > 0 or all_tensors_equal
                    one_plus_dim_tensor_dtype = utils.get_higher_dtype(
                        one_plus_dim_tensor_dtype, _dtype
                    )

        # Prefers dtype of tensors with one or more dimensions
        if one_plus_dim_tensor_dtype is not None:
            return one_plus_dim_tensor_dtype

        return zero_dim_tensor_dtype

    if highest_type is float:
        result_dtype = _find_highest_dtype_filtered(args, utils.is_float_dtype)
        result_dtype = (
            torch.get_default_dtype() if result_dtype is None else result_dtype
        )
    elif highest_type is complex:
        # NOTE: complex x float type promotion is incorrectly implemented in PyTorch today
        # it will treat zero dim and non-zero-dim float and complex tensors equally
        # unless there's a non-zero-dim complex tensor
        # the following captures this oddity
        has_one_plus_dim_complex_tensor = False
        for x in args:
            if (
                isinstance(x, TensorLike)
                and x.ndim > 0
                and utils.is_complex_dtype(x.dtype)
            ):
                has_one_plus_dim_complex_tensor = True
                break

        if has_one_plus_dim_complex_tensor:
            result_dtype = _find_highest_dtype_filtered(
                args,
                lambda x: utils.is_float_dtype(x) or utils.is_complex_dtype(x),
                float_as_complex=True,
            )
        else:
            # no complex tensors of rank 1+
            # NOTE: bugged case where all tensors are equal
            result_dtype = _find_highest_dtype_filtered(
                args,
                lambda x: utils.is_float_dtype(x) or utils.is_complex_dtype(x),
                float_as_complex=True,
                all_tensors_equal=True,
            )

        if result_dtype is None:
            result_dtype = utils.corresponding_complex_dtype(torch.get_default_dtype())
    elif highest_type is int:
        result_dtype = _find_highest_dtype_filtered(args, utils.is_integer_dtype)
        result_dtype = torch.long if result_dtype is None else result_dtype
    else:
        # highest_type is bool
        result_dtype = torch.bool

    if type_promotion is ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT:
        return result_dtype, result_dtype
    elif type_promotion is ELEMENTWISE_TYPE_PROMOTION_KIND.OP_MATH:
        return _get_computation_dtype(result_dtype), result_dtype
    elif type_promotion is ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
        if utils.is_integer_dtype(result_dtype) or utils.is_boolean_dtype(result_dtype):
            result_dtype = torch.get_default_dtype()
        return _get_computation_dtype(result_dtype), result_dtype
    elif type_promotion is ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
        if utils.is_complex_dtype(result_dtype):
            # Note: computation still occurs in complex
            return _get_computation_dtype(result_dtype), utils.corresponding_real_dtype(
                result_dtype
            )
        return _get_computation_dtype(result_dtype), result_dtype
    elif type_promotion is ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG:
        if utils.is_boolean_dtype(result_dtype):
            return torch.long, torch.long
        return result_dtype, result_dtype
    elif type_promotion is ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        return result_dtype, torch.bool
    else:
        raise ValueError("Unknown type promotion kind {0}".format(str(type_promotion)))


def _broadcast_shapes(*_shapes):
    shapes = tuple(filter(lambda x: x is not None, _shapes))

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


def broadcast(*args):
    # Computes common shape
    common_shape = _broadcast_shapes(
        *map(lambda t: t.shape if isinstance(t, TensorLike) else None, args)
    )

    def _maybe_broadcast(x, shape):
        if x is None:
            return None
        elif isinstance(x, Number):
            return x
        elif isinstance(x, TensorLike):
            common_rank = len(common_shape) + 1
            start = common_rank - (len(x.shape) + 1)
            dims = tuple(range(start, len(x.shape) + start))

            # TODO: add a pass to remove unnecessary broadcast_in_dim calls
            return prims.broadcast_in_dim(x, common_shape, dims)
        else:
            raise RuntimeError(
                "Unexpected type when broadcasting: " + str(type(x)) + "!"
            )

    return tuple(map(lambda x: _maybe_broadcast(x, common_shape), args))


# TODO: implement ref with safe casting
def _convert_dtype(*args, dtype: torch.dtype):
    def _convert(x):
        if isinstance(x, TensorLike):
            if x.dtype != dtype:
                return prims.convert_element_type(x, dtype)
        elif isinstance(x, Number):
            typ = utils.dtype_to_type(dtype)
            return typ(x)
        return x

    return tuple(map(lambda x: _convert(x), args))


def _unwrap_cpu_scalars(a, b):
    if type(a) is torch.Tensor and a.device.type == "cpu" and len(a.shape) == 0:
        a = a.item()
    elif type(b) is torch.Tensor and b.device.type == "cpu" and len(b.shape) == 0:
        b = b.item()
    return a, b


# TODO: handle tuples of tensors
def _maybe_resize_out(out: TensorLikeType, shape):
    if out.numel() == 0:
        return prims.resize(out, shape)

    if out.numel() != reduce(operator.mul, shape, 1):
        msg = (
            "An output with one or more elements was resized since it had shape {0} "
            "which does not match the required output shape {1}. "
            "This behavior is deprecated, and in a future PyTorch release outputs will not "
            "be resized unless they have zero elements. "
            "You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0).".format(
                str(out.shape), str(shape)
            )
        )
        warnings.warn(msg)
        return prims.resize(out, shape)

    return out


# Utilities should come BEFORE this import
from torch._decomp import register_decomposition


#
# Elementwise unary references
#

infer_aten_op = object()

def _make_elementwise_unary_reference(
    prim: Callable, *, type_promotion, aten_op=infer_aten_op
) -> Callable:
    def _ref(a: Tensor, *, out: Optional[Tensor] = None) -> Tensor:

        assert isinstance(a, TensorLike)
        assert out is None or isinstance(out, TensorLike)

        computation_dtype, result_dtype = _elementwise_dtypes(
            a, type_promotion=type_promotion
        )
        (a,) = _convert_dtype(a, dtype=computation_dtype)

        result = prim(a)

        if type_promotion is not ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT:
            (result,) = _convert_dtype(result, dtype=result_dtype)

        # TODO: refactor out handling to a generic wrapper
        if out is not None:
            out = _maybe_resize_out(out, result.shape)
            return copy_to(out, result, allow_cross_device=False)  # type: ignore[arg-type]

        return result

    if aten_op is infer_aten_op:
        aten_op = getattr(torch.ops.aten, prim.__name__)
    if aten_op is not None:
        register_decomposition(aten_op)(_ref)

    return _ref


abs = _make_elementwise_unary_reference(
    prims.abs, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
)

acos = _make_elementwise_unary_reference(
    prims.acos, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

acosh = _make_elementwise_unary_reference(
    prims.acosh, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

asin = _make_elementwise_unary_reference(
    prims.asin, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

atan = _make_elementwise_unary_reference(
    prims.atan, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

ceil = _make_elementwise_unary_reference(
    prims.ceil, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)

cos = _make_elementwise_unary_reference(
    prims.cos, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

cosh = _make_elementwise_unary_reference(
    prims.cosh, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

digamma = _make_elementwise_unary_reference(
    prims.digamma, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

erf = _make_elementwise_unary_reference(
    prims.erf, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

erfinv = _make_elementwise_unary_reference(
    prims.erf_inv,
    type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    aten_op=torch.ops.aten.erfinv,  # prim/aten name mismatch
)

erfc = _make_elementwise_unary_reference(
    prims.erfc, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

exp = _make_elementwise_unary_reference(
    prims.exp, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

expm1 = _make_elementwise_unary_reference(
    prims.expm1, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

floor = _make_elementwise_unary_reference(
    prims.floor, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)

isfinite = _make_elementwise_unary_reference(
    prims.is_finite,
    type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=None,  # CompositeImplicitAutograd
)


def _isnan(a: Tensor) -> Tensor:
    return prims.ne(a, a)


isnan = _make_elementwise_unary_reference(
    _isnan,
    type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=torch.ops.aten.isnan,  # prim/aten name mismatch
)

lgamma = _make_elementwise_unary_reference(
    prims.lgamma, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

log = _make_elementwise_unary_reference(
    prims.log, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

log1p = _make_elementwise_unary_reference(
    prims.log1p, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

neg = _make_elementwise_unary_reference(
    prims.neg, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)

reciprocal = _make_elementwise_unary_reference(
    prims.reciprocal, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

# TODO: round takes additional kwargs
round = _make_elementwise_unary_reference(
    prims.round,
    type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=None,  # TODO: this does need a decomp, but kwarg handling is needed
)

sign = _make_elementwise_unary_reference(
    prims.sign, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)

sin = _make_elementwise_unary_reference(
    prims.sin, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

sinh = _make_elementwise_unary_reference(
    prims.sinh, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

sqrt = _make_elementwise_unary_reference(
    prims.sqrt, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

square = _make_elementwise_unary_reference(
    prims.square,
    type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG,
    aten_op=None,  # CompositeImplicitAutograd
)

tan = _make_elementwise_unary_reference(
    prims.tan, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)


def _make_elementwise_binary_reference(prim: Callable, *, type_promotion, wrap_scalars=False, aten_op=infer_aten_op) -> Callable:
    def _ref(
        a: Union[Tensor, NumberType],
        b: Union[Tensor, NumberType],
        *,
        out: Optional[Tensor] = None
    ) -> Tensor:
        assert isinstance(a, (TensorLike, Number))
        assert isinstance(b, (TensorLike, Number))
        assert out is None or isinstance(out, TensorLike)

        # Special-cases Number x Number case
        if isinstance(a, Number) and isinstance(b, Number):
            if wrap_scalars:
                a, b = utils.wrap_scalars(a, b)
            else:
                raise RuntimeError("got two scalar arguments, while expected at least one TensorLike")

        # Handles type promotion
        computation_dtype, result_dtype = _elementwise_dtypes(
            a, b, type_promotion=type_promotion
        )
        a, b = _convert_dtype(a, b, dtype=computation_dtype)

        # Special case CPU scalar tensors to be eligible for device transfer
        if wrap_scalars:
            a, b = _unwrap_cpu_scalars(a, b)

        # Broadcasting
        a, b = broadcast(a, b)

        result = prim(a, b)

        if type_promotion is not ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT:
            (result,) = _convert_dtype(result, dtype=result_dtype)

        if out is not None:
            out = _maybe_resize_out(out, result.shape)
            return copy_to(out, result, allow_cross_device=False)  # type: ignore[arg-type]

        return result

    if aten_op is infer_aten_op:
        aten_op = getattr(torch.ops.aten, prim.__name__)
    if aten_op is not None:
        register_decomposition(aten_op)(_ref)

    return _ref


# Add is implemented specially because it has an alpha argument and
#   is decomposed into multiple prims
def add(
    a: Union[Tensor, NumberType],
    b: Union[Tensor, NumberType],
    *,
    alpha: Optional[NumberType] = None,
    out: Optional[Tensor] = None
):
    """
    Reference implementation of torch.add
    """

    # Type checks
    assert isinstance(a, (TensorLike, Number))
    assert isinstance(b, (TensorLike, Number))
    assert out is None or isinstance(out, TensorLike)
    assert alpha is None or isinstance(alpha, Number)

    # Special-cases Number x Number case
    if isinstance(a, Number) and isinstance(b, Number):
        a, b = utils.wrap_scalars(a, b)

    computation_dtype, result_dtype = _elementwise_dtypes(
        a, b, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.OP_MATH
    )
    a, b = _convert_dtype(a, b, dtype=computation_dtype)

    a, b = _unwrap_cpu_scalars(a, b)

    a, b = broadcast(a, b)

    if alpha is not None:
        alpha_promotion_type = utils.dtype_to_type(computation_dtype)
        assert utils.is_lesser_type(type(alpha), alpha_promotion_type) or (
            computation_dtype is torch.bool and type(alpha) is int
        )
        b = prims.mul(b, alpha_promotion_type(alpha))

    result = prims.add(a, b)

    (result,) = _convert_dtype(result, dtype=result_dtype)

    if out is not None:
        out = _maybe_resize_out(out, result.shape)
        return copy_to(out, result, allow_cross_device=False)  # type: ignore[arg-type]

    return result


# TODO: add docstring
atan2 = _make_elementwise_binary_reference(
    prims.atan2, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

# TODO: add docstring
bitwise_and = _make_elementwise_binary_reference(
    prims.bitwise_and,
    type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: add docstring
bitwise_left_shift = _make_elementwise_binary_reference(
    prims.shift_left,
    type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.bitwise_left_shift,  # prim/aten name mismatch
)

# TODO: add docstring
bitwise_or = _make_elementwise_binary_reference(
    prims.bitwise_or, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)

# TODO: add docstring
bitwise_right_shift = _make_elementwise_binary_reference(
    prims.shift_right_arithmetic,
    type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.bitwise_right_shift,  # prim/aten name mismatch
)

# TODO: add docstring
bitwise_xor = _make_elementwise_binary_reference(
    prims.bitwise_xor, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)

# TODO: add docstring
# complex =  _make_elementwise_binary_reference(prims.complex, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)

# TODO: add docstring
eq = _make_elementwise_binary_reference(
    prims.eq, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
)

# TODO: add docstring
# Float power has its own implementation because it has unique type promotion.
# NB: aten_op not registered because CompositeExplicitAutograd
def float_power(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    out: Optional[TensorLikeType] = None,
) -> Tensor:

    assert isinstance(a, (TensorLike, Number))
    assert isinstance(b, (TensorLike, Number))
    assert out is None or isinstance(out, TensorLike)

    # Handles type promotion
    dtype = utils.get_higher_dtype(a, b)
    assert dtype is not None
    if utils.is_complex_dtype(dtype):
        dtype = torch.complex128
    else:
        dtype = torch.float64

    a, b = _convert_dtype(a, b, dtype=dtype)

    # Broadcasting
    a, b = broadcast(a, b)

    result = prims.pow(a, b)

    if out is not None:
        out = _maybe_resize_out(out, result.shape)
        return copy_to(out, result, allow_cross_device=False)  # type: ignore[arg-type]

    return result


# TODO: add docstring
ge = _make_elementwise_binary_reference(
    prims.ge, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
)

# TODO: add docstring
gt = _make_elementwise_binary_reference(
    prims.gt, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
)

igamma = _make_elementwise_binary_reference(
    prims.igamma, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

igammac = _make_elementwise_binary_reference(
    prims.igammac, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

# TODO: add docstring
le = _make_elementwise_binary_reference(
    prims.le, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
)

# TODO: add docstring
lt = _make_elementwise_binary_reference(
    prims.lt, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
)

# TODO: add docstring
maximum = _make_elementwise_binary_reference(
    prims.max,
    type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.maximum,  # prim/aten name mismatch
)

# TODO: add docstring
minimum = _make_elementwise_binary_reference(
    prims.min,
    type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=torch.ops.aten.minimum,  # prim/aten name mismatch
)

# TODO: add docstring
mul = _make_elementwise_binary_reference(
    prims.mul, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.OP_MATH,
    wrap_scalars=True
)

# TODO: add docstring
ne = _make_elementwise_binary_reference(
    prims.ne, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
)

# TODO: add docstring
nextafter = _make_elementwise_binary_reference(
    prims.nextafter, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)

# TODO: add docstring
pow = _make_elementwise_binary_reference(
    prims.pow, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
)

# TODO: add docstring
# Sub is implemented specially because it has an alpha argument and
#   is decomposed into multiple prims.
# TODO: consider refactoring this with add impl
@register_decomposition(torch.ops.aten.sub)
def sub(
    a: Union[Tensor, NumberType],
    b: Union[Tensor, NumberType],
    *,
    alpha: Optional[NumberType] = None,
    out: Optional[Tensor] = None
):
    """
    Reference implementation of torch.sub
    """

    # Type checks
    assert isinstance(a, (TensorLike, Number))
    assert isinstance(b, (TensorLike, Number))
    assert out is None or isinstance(out, TensorLike)
    assert alpha is None or isinstance(alpha, Number)

    # Special-cases Number x Number case
    if isinstance(a, Number) and isinstance(b, Number):
        a, b = utils.wrap_scalars(a, b)

    computation_dtype, result_dtype = _elementwise_dtypes(
        a, b, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.OP_MATH
    )
    a, b = _convert_dtype(a, b, dtype=computation_dtype)

    a, b = _unwrap_cpu_scalars(a, b)

    a, b = broadcast(a, b)

    if alpha is not None:
        alpha_promotion_type = utils.dtype_to_type(computation_dtype)
        assert utils.is_lesser_type(type(alpha), alpha_promotion_type) or (
            computation_dtype is torch.bool and type(alpha) is int
        )
        b = prims.mul(b, alpha_promotion_type(alpha))

    result = prims.sub(a, b)

    (result,) = _convert_dtype(result, dtype=result_dtype)

    if out is not None:
        out = _maybe_resize_out(out, result.shape)
        return copy_to(out, result, allow_cross_device=False)  # type: ignore[arg-type]

    return result


# TODO: add docstring
true_divide = _make_elementwise_binary_reference(
    prims.div, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    wrap_scalars=True,
    aten_op=None,  # CompositeImplicitAutograd
)

#
# Conditional references
#

# https://pytorch.org/docs/stable/generated/torch.where.html
# TODO: implement alternate where
@register_decomposition(torch.ops.aten.where)
def where(
    pred: Tensor,
    a: Optional[Tensor] = None,
    b: Optional[Tensor] = None,
    *,
    out: Optional[Tensor] = None
):
    """ """

    if a is None:
        raise NotImplementedError

    # Type checking
    assert isinstance(pred, TensorLike)
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)
    if out is not None:
        assert isinstance(out, TensorLike)

    _, result_dtype = _elementwise_dtypes(
        a, b, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    a, b = _convert_dtype(a, b, dtype=result_dtype)

    pred, a, b = broadcast(pred, a, b)

    result = prims.select(pred, a, b)

    if out is not None:
        out = _maybe_resize_out(out, result.shape)
        return copy_to(out, result, allow_cross_device=False)  # type: ignore[arg-type]

    return result


#
# Data Movement References
#


def copy_to(a: Tensor, b: Tensor, *, allow_cross_device=True):
    if not allow_cross_device and a.device != b.device:
        msg = "Attempting to copy from device {0} to device {1}, but cross-device copies are not allowed!".format(
            b.device, a.device
        )
        raise RuntimeError(msg)

    return prims.copy_to(a, b)


#
# Reduction references
#


def _reduction(
    a: Tensor,
    prim: Callable,
    *,
    has_identity: bool = True,
    accepts_dim_tuple: bool = True,  # to handle min/argmin that accept single dim only
    dims: Optional[DimsType] = None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,  # should be specified for ops that support it
    out: Optional[Tensor] = None,
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND
):  # it is usually SAME, but I want
    # ref writers to actually think about what to put here
    assert isinstance(a, TensorLike)
    if out is not None:
        assert isinstance(out, TensorLike)
        if dtype is not None:
            # TODO - this is true for eager mode currently, but it's wrong behavior for complex norms
            if dtype != out.dtype:
                raise RuntimeError(
                    "dtype argument and out dtype must match in reduction"
                )
    if not accepts_dim_tuple:
        assert dims is None or isinstance(dims, int)
    if isinstance(dims, int):
        dims = (dims,)  # type: ignore[assignment]
    dims = utils.reduction_dims(a.shape, dims)
    if not has_identity:
        valid_shape = all(a.shape[i] for i in range(a.ndim) if i in dims)
        if not valid_shape:
            raise RuntimeError("reducing over zero-size dimension for reduction operation without identity")
    # even though some reductions, like amin or amax, don't strictly require type promotion,
    # all the math ops (including comparisons) are still defined only for a computation type,
    # so promotion will still happen. We are doing it explicitly here
    inp_dtype = dtype if dtype is not None else a.dtype
    computation_dtype = _get_computation_dtype(inp_dtype)
    a_converted = prims.convert_element_type(a, computation_dtype)
    result = prim(a_converted, dims)

    if keepdims:
        output_shape = [a.shape[i] if i not in dims else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dims]
        result = prims.broadcast_in_dim(result, output_shape, broadcast_dims)
    if out is not None:
        if dtype is None:
            if output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.SAME:
                if out.dtype != a.dtype:
                    raise RuntimeError(
                        "Expected the dtype for input and out to match"
                    )
            elif output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.ALWAYS_BOOL:
                if out.dtype != torch.bool:
                    raise RuntimeError(
                        "Expected the dtype for input and out to match"
                    )
        out = _maybe_resize_out(out, result.shape)
        return copy_to(out, result, allow_cross_device=False)  # type: ignore[arg-type]

    if output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.SAME:
        result_dtype = dtype if dtype else a.dtype
        result = prims.convert_element_type(result, result_dtype)
    return result


# TODO: register decomp after stride logic is fixed
def sum(
    a: Tensor,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    dtype=None,
    out: Optional[Tensor] = None
):
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

def amin(
    a: Tensor,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None
):
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

def amax(
    a: Tensor,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None
):
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


def cat(
    tensors: TensorSequenceType, dim: int = 0, out: TensorLikeType = None
) -> TensorLikeType:
    if len(tensors) == 0:
        msg = "cat expects at least one tensor, but received zero!"
        raise ValueError(msg)

    _dim = utils.canonicalize_dims(tensors[0].ndim, dim)
    dtype, _ = _elementwise_dtypes(
        *tensors, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )

    _tensors = _convert_dtype(*tensors, dtype=dtype)
    result = prims.concatenate(_tensors, _dim)

    if out is not None:
        out = _maybe_resize_out(out, result.shape)
        return copy_to(out, result, allow_cross_device=False)  # type: ignore[arg-type]

    return result


def permute(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType:
    _permutation = utils.canonicalize_dims(a.ndim, dims)
    return prims.transpose(a, _permutation)


# Note: does not work with TensorMetas because of data-dependent control-flow
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
        if indices_or_sections.device != torch.device("cpu"):
            msg = "tensor_split: if indices_or_sections is a tensor it must be on the CPU, but received one on {0}".format(
                indices_or_sections.device
            )
            raise ValueError(msg)
        if indices_or_sections.dtype != torch.long:
            msg = "tensor_split: if indices_or_sections is a tensor it must have long dtype, "
            " but received one with dtype {0}".format(indices_or_sections.dtype)
            raise ValueError(msg)

    # Case 0 -- indices_or_sections is an integer or a scalar tensor n and a is split along dim into n parts of equal-ish length
    if isinstance(indices_or_sections, int) or (
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


def transpose(a: TensorLikeType, dim0: int, dim1: int) -> TensorLikeType:
    _dim0, _dim1 = utils.canonicalize_dims(a.ndim, (dim0, dim1))  # type: ignore[misc]

    if a.ndim <= 1:
        return a

    _permutation = list(range(0, a.ndim))
    _permutation[_dim0] = _dim1
    _permutation[_dim1] = _dim0
    return prims.transpose(a, _permutation)


# Aliases for transpose
swap_axes = transpose
