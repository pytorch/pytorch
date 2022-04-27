import torch
from torch._C import _add_docstr  # type: ignore[attr-defined]

import torch._prims as prims
import torch._prims.utils as utils
from torch._prims import TensorLike as TensorLike
from torch._prims.utils import DimsType

from functools import reduce
from enum import Enum
from numbers import Number, Complex
from typing import Sequence, Optional, Union, Callable, List

# Experimental module containing prototype Python references for existing
#   PyTorch operations.

all = [
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
    # 'igamma',
    # 'igammac',
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
    "where",
    #
    # Data conversion and movement references
    #
    "copy_to",
    #
    # Reduction ops
    #
    "sum",
]

Tensor = torch.Tensor


class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    INT_TO_FLOAT = (1,)
    ALWAYS_BOOL = (2,)
    OP_MATH = 3


class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME = (0,)
    SAME_OR_REAL = (1,)  # for complex types outputs corresponding real type
    OP_MATH = (2,)  # keep output in opmath type, needed for mean
    ALWAYS_BOOL = (3,)


# Maps datatypes to their computation types for elementwise operations
_computation_dtype_map = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
}


def _get_computation_dtype(dtype: torch.dtype):
    return _computation_dtype_map.get(dtype, dtype)


# TODO: document type promotion kinds
def _elementwise_dtypes(*_args, type_promotion: ELEMENTWISE_TYPE_PROMOTION_KIND):
    """
    Computes the computation and result dtypes for elementwise type promotion
    on the given arguments and with the given elementwise type promotion kind.

    Elementwise type promotion first decides which of four ordered types to use:

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
          then the result dtype is the dtype corresponding to the selected type
      - if no tensor with one or dimensions' dtype has the same corresponding type as the one
          selected, then the result dtype is the highest dtype among all tensors
      - if the first two cases do not apply, the result dtype is the highest dtype among
          all tensors with one or more dimensions

    The computation dtype is usually the result dtype, except for float16 and bfloat16, where
    the computation dtype is float32, and complex32, where the computation dtype is complex64.
    """

    args = tuple(filter(lambda x: x is not None, _args))

    # Type checking
    for arg in args:
        assert isinstance(arg, (Number, TensorLike))

    # Determines datatypes for each category
    scalar_args = filter(lambda x: isinstance(x, Number), args)
    scalar_type = reduce(
        lambda acc, x: utils.get_higher_type(acc, type(x)), scalar_args, bool  # type: ignore[arg-type, return-value]
    )

    scalar_tensors = filter(lambda t: isinstance(t, TensorLike) and t.ndim == 0, args)
    scalar_tensor_dtype = reduce(
        utils.get_higher_dtype, (t.dtype for t in scalar_tensors), torch.bool
    )
    scalar_tensor_type = utils.dtype_to_type(scalar_tensor_dtype)

    nonscalar_tensors = filter(
        lambda t: isinstance(t, TensorLike) and t.ndim != 0, args
    )
    nonscalar_tensor_dtype = reduce(
        utils.get_higher_dtype, (t.dtype for t in nonscalar_tensors), torch.bool
    )
    nonscalar_tensor_type = utils.dtype_to_type(nonscalar_tensor_dtype)

    typ = reduce(
        utils.get_higher_type, (scalar_type, scalar_tensor_type, nonscalar_tensor_type)
    )

    if nonscalar_tensor_type is typ:
        dtype = nonscalar_tensor_dtype
    elif scalar_tensor_type is typ:
        dtype = scalar_tensor_dtype
    else:
        # scalar type kind -> default torch dtype mapping
        if typ is bool:
            dtype = torch.bool
        elif typ is int:
            dtype = torch.int64
        elif typ is float:
            dtype = torch.get_default_dtype()
        else:
            # typ is complex
            dtype = (
                torch.complex128
                if torch.get_default_dtype() is torch.float64
                else torch.complex64
            )

    if type_promotion is ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT and (
        utils.is_boolean_dtype(dtype) or utils.is_integer_dtype(dtype)
    ):
        return torch.get_default_dtype(), torch.get_default_dtype()

    if type_promotion is ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        return dtype, torch.bool

    if type_promotion is ELEMENTWISE_TYPE_PROMOTION_KIND.OP_MATH:
        return _get_computation_dtype(dtype), dtype

    # DEFAULT type promotion
    return dtype, dtype


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
            return prims.convert_element_type(x, dtype)
        elif isinstance(x, Number):
            typ = utils.dtype_to_type(dtype)
            return typ(x)
        return x

    return tuple(map(lambda x: _convert(x), args))


# TODO: handle tuples of tensors
def _maybe_resize_out(out: Tensor, shape):
    if out.numel() == 0:
        return prims.resize(out, shape)

    return out


def _make_elementwise_binary_reference(prim: Callable, *, type_promotion) -> Callable:
    def _ref(
        a: Union[Tensor, Number], b: Union[Tensor, Number], out: Optional[Tensor] = None
    ) -> Tensor:
        assert isinstance(a, (TensorLike, Number))
        assert isinstance(b, (TensorLike, Number))
        assert out is None or isinstance(out, TensorLike)

        # Special-cases Number x Number case
        if isinstance(a, Number) and isinstance(b, Number):
            a, b = utils.wrap_scalars(a, b)

        # Handles type promotion
        computation_dtype, result_dtype = _elementwise_dtypes(
            a, b, type_promotion=type_promotion
        )
        a, b = _convert_dtype(a, b, dtype=computation_dtype)

        # Broadcasting
        a, b = broadcast(a, b)

        result = prim(a, b)

        if type_promotion is not ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT:
            (result,) = _convert_dtype(result, dtype=result_dtype)

        if out is not None:
            out = _maybe_resize_out(out, result.shape)
            return copy_to(out, result, allow_cross_device=False)  # type: ignore[arg-type]

        return result

    return _ref


# Add is implemented specially because it has an alpha argument and
#   is decomposed into multiple prims
def add(
    a: Union[Tensor, Number],
    b: Union[Tensor, Number],
    *,
    alpha: Optional[Number] = None,
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
    prims.bitwise_and, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)

# TODO: add docstring
bitwise_left_shift = _make_elementwise_binary_reference(
    prims.shift_left, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)

# TODO: add docstring
bitwise_or = _make_elementwise_binary_reference(
    prims.bitwise_or, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)

# TODO: add docstring
bitwise_right_shift = _make_elementwise_binary_reference(
    prims.shift_right_arithmetic, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
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
def float_power(
    a: Union[Tensor, Number], b: Union[Tensor, Number], out: Optional[Tensor] = None
) -> Tensor:

    assert isinstance(a, (Tensor, Number))
    assert isinstance(b, (Tensor, Number))
    assert out is None or isinstance(out, TensorLike)

    # Special-cases Number x Number case
    if isinstance(a, Number) and isinstance(b, Number):
        a, b = utils.wrap_scalars(a, b)

    # Handles type promotion
    dtype = utils.get_higher_dtype(a, b)
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

# TODO: add docstring
le = _make_elementwise_binary_reference(
    prims.lt, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
)

# TODO: add docstring
lt = _make_elementwise_binary_reference(
    prims.lt, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
)

# TODO: add docstring
maximum = _make_elementwise_binary_reference(
    prims.max, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)

# TODO: add docstring
minimum = _make_elementwise_binary_reference(
    prims.min, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)

# TODO: add docstring
mul = _make_elementwise_binary_reference(
    prims.mul, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.OP_MATH
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
    prims.pow, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.OP_MATH
)

# TODO: add docstring
# Sub is implemented specially because it has an alpha argument and
#   is decomposed into multiple prims.
# TODO: consider refactoring this with add impl
def sub(
    a: Union[Tensor, Number],
    b: Union[Tensor, Number],
    *,
    alpha: Optional[Number] = None,
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
    prims.div, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

#
# Conditional references
#

# https://pytorch.org/docs/stable/generated/torch.where.html
# TODO: implement alternate where
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
            assert (
                dtype == out.dtype
            ), "dtype argument and out dtype must match in reduction"
    if not accepts_dim_tuple:
        assert dims is None or isinstance(dims, int)
    if isinstance(dims, int):
        dims = (dims,)  # type: ignore[assignment]
    dims = utils.reduction_dims(a.shape, dims)
    if not has_identity:
        valid_shape = all(a.shape[i] for i in range(a.ndim) if i in dims)  # type: ignore[operator]
        assert (
            valid_shape
        ), "reducing over zero-size dimension for reduction operation without identity"
    # even though some reductions, like amin or amax, don't strictly require type promotion,
    # all the math ops (including comparisons) are still defined only for a computation type,
    # so promotion will still happen. We are doing it explicitly here
    inp_dtype = dtype if dtype is not None else a.dtype
    computation_dtype = _get_computation_dtype(inp_dtype)
    a_converted = prims.convert_element_type(a, computation_dtype)
    result = prim(a_converted, dims)

    if keepdims:
        output_shape = [a.shape[i] if i in dims else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dims]
        result = prims.broadcast_in_dim(result, output_shape, broadcast_dims)
    if out is not None:
        if dtype is None:
            if output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.SAME:
                assert (
                    out.dtype == a.dtype
                ), "out dtype and output type of reduction must match"
            elif output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.ALWAYS_BOOL:
                assert (
                    out.dtype == torch.bool
                ), "out dtype and output type of reduction must match"
        out = _maybe_resize_out(out, result.shape)
        return copy_to(out, result, allow_cross_device=False)  # type: ignore[arg-type]

    if output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.SAME:
        result = prims.convert_element_type(result, a.dtype)
    return result


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

    return _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )
