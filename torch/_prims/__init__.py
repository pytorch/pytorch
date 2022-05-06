import torch
from torch import Tensor

import torch._prims.utils as utils
from torch._prims.utils import TensorLike, TensorLikeType, TensorMeta, ShapeType
from torch.overrides import has_torch_function, handle_torch_function

from typing import Sequence, Optional, Union, Callable, List, Tuple
from numbers import Number
from functools import reduce
from enum import Enum

# Experimental module containing prototype "primitive" operations.

__all__ = [
    #
    # Common datastructures and helpers
    #
    "RETURN_TYPE",
    #
    # Elementwise unary prims
    #
    "abs",
    "acos",
    "acosh",
    "asin",
    "atan",
    "cos",
    "cosh",
    "bessel_i0e",
    "bessel_i1e",
    "cbrt",
    "ceil",
    "digamma",
    "erf",
    "erf_inv",
    "erfc",
    "exp",
    "expm1",
    "floor",
    "igamma",
    "igammac",
    "is_finite",
    "lgamma",
    "log",
    "log1p",
    "neg",
    "reciprocal",
    "round",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
    #
    # Elementwise binary prims
    #
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    # 'complex',  # needs custom meta
    "div",
    "eq",
    "ge",
    "gt",
    "le",
    "lt",
    "max",
    "min",
    "mul",
    "ne",
    "nextafter",
    "pow",
    "rsqrt",
    "shift_left",
    "shift_right_arithmetic",
    "shift_right_logical",  # not implemented
    #
    # View prims
    #
    "broadcast_in_dim",
    "collapse_view",
    "split_dim",
    "squeeze",
    #
    # Shape prims
    #
    "collapse",
    "concatenate",
    "reshape",
    #
    # Conditional prims
    #
    "select",
    #
    # Data conversion and movement prims
    #
    "convert_element_type",
    "device_put",
    #
    # Inplace prims
    #
    "copy_to",
    "resize",
]

#
# Common datastructures and helpers
#

# Describes the return type of the primitive:
#
#   - NEW, a new tensor is created
#   - VIEW, a view of an input tensor is returned
#   - INPLACE, one or more input tensors is modified
#
# these descriptors are mututally exclusive and exhaustive.
class RETURN_TYPE(Enum):
    NEW = (0,)
    VIEW = (1,)
    INPLACE = (2,)


def _make_prim(
    *, meta: Callable, impl_aten: Callable, return_type: RETURN_TYPE, doc: str
):
    """
    Creates a primitive operation.

    """

    def _prim(*args, **kwargs):
        # TODO: allow dispatch to be overridden here
        if has_torch_function(args):
            return handle_torch_function(_prim, args, *args, **kwargs)

        # always run the meta function because aten implementation will
        # typically accept more inputs (e.g., it will do promotion and
        # broadcasting) which we want to reject
        meta(*args, **kwargs)
        return impl_aten(*args, **kwargs)

    _prim.__doc__ = doc
    _prim.meta = meta  # type: ignore[attr-defined]
    _prim.return_type = return_type  # type: ignore[attr-defined]

    return _prim


def _elementwise_meta(*args):
    """
    Meta function for elementwise operations that produce outputs in the same dtype
    as their inputs.

    Stride logic is currently incorrect.
    """

    assert len(args) > 0

    utils.check_same_device(*args, allow_scalars=True)
    utils.check_same_shape(*args)
    utils.check_same_dtype(*args)

    strides = None
    tensor = None
    number = None
    for arg in args:
        if isinstance(arg, TensorLike):
            if strides is None:
                strides = arg.stride()

            if tensor is None:
                tensor = arg

            if arg.stride() != strides:
                return TensorMeta(
                    arg, strides=utils.make_contiguous_strides_for(arg.shape)
                )
        elif isinstance(arg, Number):
            if number is None:
                number = arg

    # TODO: fix strides
    if tensor is not None:
        if 0 in tensor.stride() and tensor.numel() > 0:
            return TensorMeta(
                tensor, strides=utils.make_contiguous_strides_for(tensor.shape)
            )
        else:
            return TensorMeta(tensor)

    return TensorMeta(number)


def _make_elementwise_unary_prim(impl_aten: Callable, doc: str):
    """
    Creates an elementwise unary prim.
    """

    return _make_prim(
        meta=_elementwise_meta,
        impl_aten=impl_aten,
        return_type=RETURN_TYPE.NEW,
        doc=doc,
    )


def _make_elementwise_binary_prim(impl_aten: Callable, doc: str):
    """
    Creates an elementwise binary prim.
    """

    return _make_prim(
        meta=_elementwise_meta,
        impl_aten=impl_aten,
        return_type=RETURN_TYPE.NEW,
        doc=doc,
    )


def _not_impl(*args, **kwargs):
    raise NotImplementedError


#
# Elementwise unary operations
#

abs = _make_elementwise_unary_prim(
    impl_aten=torch.abs,
    doc="",
)

acos = _make_elementwise_unary_prim(
    impl_aten=torch.acos,
    doc="",
)

acosh = _make_elementwise_unary_prim(
    impl_aten=torch.acosh,
    doc="",
)

asin = _make_elementwise_unary_prim(
    impl_aten=torch.asin,
    doc="",
)

atan = _make_elementwise_unary_prim(
    impl_aten=torch.atan,
    doc="",
)

cos = _make_elementwise_unary_prim(
    impl_aten=torch.cos,
    doc="",
)

cosh = _make_elementwise_unary_prim(
    impl_aten=torch.cosh,
    doc="",
)

bessel_i0e = _make_elementwise_unary_prim(
    impl_aten=torch.special.i0e,
    doc="",
)

bessel_i1e = _make_elementwise_unary_prim(
    impl_aten=torch.special.i1e,
    doc="",
)


def _cbrt_aten(a: torch.Tensor):
    return pow(a, (1 / 3))


cbrt = _make_elementwise_unary_prim(
    impl_aten=_cbrt_aten,
    doc="",
)

ceil = _make_elementwise_unary_prim(
    impl_aten=torch.ceil,
    doc="",
)

digamma = _make_elementwise_unary_prim(
    impl_aten=torch.digamma,
    doc="",
)

erf = _make_elementwise_unary_prim(
    impl_aten=torch.erf,
    doc="",
)

erf_inv = _make_elementwise_unary_prim(
    impl_aten=torch.special.erfinv,
    doc="",
)

erfc = _make_elementwise_unary_prim(
    impl_aten=torch.special.erfc,
    doc="",
)

exp = _make_elementwise_unary_prim(
    impl_aten=torch.exp,
    doc="",
)

expm1 = _make_elementwise_unary_prim(
    impl_aten=torch.special.expm1,
    doc="",
)

floor = _make_elementwise_unary_prim(
    impl_aten=torch.floor,
    doc="",
)

igamma = _make_elementwise_unary_prim(
    impl_aten=torch.special.gammainc,
    doc="",
)

igammac = _make_elementwise_unary_prim(
    impl_aten=torch.special.gammaincc,
    doc="",
)

is_finite = _make_elementwise_unary_prim(
    impl_aten=torch.isfinite,
    doc="",
)

lgamma = _make_elementwise_unary_prim(
    impl_aten=torch.lgamma,
    doc="",
)

log = _make_elementwise_unary_prim(
    impl_aten=torch.log,
    doc="",
)

log1p = _make_elementwise_unary_prim(
    impl_aten=torch.log1p,
    doc="",
)

reciprocal = _make_elementwise_unary_prim(
    impl_aten=torch.reciprocal,
    doc="",
)

neg = _make_elementwise_unary_prim(
    impl_aten=torch.neg,
    doc="",
)

round = _make_elementwise_unary_prim(
    impl_aten=torch.round,
    doc="",
)

sign = _make_elementwise_unary_prim(
    impl_aten=torch.sign,
    doc="",
)

sin = _make_elementwise_unary_prim(
    impl_aten=torch.sin,
    doc="",
)

sinh = _make_elementwise_unary_prim(
    impl_aten=torch.sinh,
    doc="",
)

sqrt = _make_elementwise_unary_prim(
    impl_aten=torch.sqrt,
    doc="",
)

square = _make_elementwise_unary_prim(
    impl_aten=torch.square,
    doc="",
)

tan = _make_elementwise_unary_prim(
    impl_aten=torch.tan,
    doc="",
)

#
# Elementwise binary operations
#

add = _make_elementwise_binary_prim(
    impl_aten=torch.add,
    doc="",
)

atan2 = _make_elementwise_binary_prim(
    impl_aten=torch.atan2,
    doc="",
)

bitwise_and = _make_elementwise_binary_prim(
    impl_aten=torch.bitwise_and,
    doc="",
)

bitwise_not = _make_elementwise_binary_prim(
    impl_aten=torch.bitwise_not,
    doc="",
)

bitwise_or = _make_elementwise_binary_prim(
    impl_aten=torch.bitwise_or,
    doc="",
)

bitwise_xor = _make_elementwise_binary_prim(
    impl_aten=torch.bitwise_xor,
    doc="",
)

# TODO: complex needs a special meta to account for its float -> complex behavior
# complex = _make_elementwise_binary_prim(
#   impl_aten=torch.complex,
#   doc="",
# )

# div prim performs truncation division on integer inputs
#   and true division for floating and complex inputs
def _div(a, b):
    if isinstance(a, (bool, int)):
        return torch.div(a, b, rounding_mode="trunc")
    return torch.true_divide(a, b)


div = _make_elementwise_binary_prim(
    impl_aten=_div,
    doc="",
)

eq = _make_elementwise_binary_prim(
    impl_aten=torch.eq,
    doc="",
)

ge = _make_elementwise_binary_prim(
    impl_aten=torch.ge,
    doc="",
)

gt = _make_elementwise_binary_prim(
    impl_aten=torch.gt,
    doc="",
)

le = _make_elementwise_binary_prim(
    impl_aten=torch.le,
    doc="",
)

lt = _make_elementwise_binary_prim(
    impl_aten=torch.lt,
    doc="",
)

max = _make_elementwise_binary_prim(
    impl_aten=torch.maximum,
    doc="",
)

min = _make_elementwise_binary_prim(
    impl_aten=torch.minimum,
    doc="",
)

mul = _make_elementwise_binary_prim(
    impl_aten=torch.mul,
    doc="",
)

ne = _make_elementwise_binary_prim(
    impl_aten=torch.ne,
    doc="",
)

nextafter = _make_elementwise_binary_prim(
    impl_aten=torch.nextafter,
    doc="",
)

pow = _make_elementwise_binary_prim(
    impl_aten=torch.pow,
    doc="",
)

rsqrt = _make_elementwise_binary_prim(
    impl_aten=torch.rsqrt,
    doc="",
)

shift_left = _make_elementwise_binary_prim(
    impl_aten=torch.bitwise_left_shift,
    doc="",
)

shift_right_arithmetic = _make_elementwise_binary_prim(
    impl_aten=torch.bitwise_right_shift,
    doc="",
)

shift_right_logical = _not_impl

sub = _make_elementwise_binary_prim(
    impl_aten=torch.sub,
    doc="",
)

#
# View operations
#
def _broadcast_in_dim_meta(
    a: TensorLikeType, shape: ShapeType, broadcast_dimensions: Sequence[int]
):
    # Type checks
    assert isinstance(a, TensorLike)
    assert isinstance(shape, Sequence)
    assert isinstance(broadcast_dimensions, Sequence)

    # every dimension must be accounted for
    assert a.ndim == len(broadcast_dimensions)

    # broadcast shape must have weakly more dimensions
    assert len(shape) >= a.ndim

    # broadcast_dimensions must be an ascending sequence
    # (no relative reordering of dims) of integers and
    # each dimension must be within the new shape
    def _greater_than_reduce(acc, x):
        assert isinstance(x, int)
        assert x > acc
        assert x < len(shape)

        return x

    reduce(lambda acc, x: _greater_than_reduce(acc, x), broadcast_dimensions, -1)

    # shape must be broadcastable to
    for idx, new_idx in enumerate(broadcast_dimensions):
        assert a.shape[idx] == 1 or a.shape[idx] == shape[new_idx]

    new_strides = []
    original_idx = 0
    for idx in range(len(shape)):
        if idx in broadcast_dimensions:
            new_strides.append(a.stride()[original_idx])
            original_idx = original_idx + 1
        else:
            new_strides.append(0)

    return TensorMeta(a, shape=shape, strides=new_strides)


def _broadcast_in_dim_aten(a, shape, broadcast_dimensions):
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = v.unsqueeze(idx)

    return v.expand(shape)


_broadcast_in_dim_doc = """
  Creates a view of t with the specified shape.

  Allows adding dimensions of any length and broadcasting
  dimensions of length one in t to any length.

  The location of the broadcast dimensions must be specified
  using the broadcast_dimensions argument. Changing the
  relative order of dimensions is not supported.
  """

broadcast_in_dim = _make_prim(
    meta=_broadcast_in_dim_meta,
    impl_aten=_broadcast_in_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_broadcast_in_dim_doc,
)


def _collapse_view_meta(a: TensorLikeType, start: int, end: int) -> TensorLikeType:
    assert isinstance(a, TensorLike)

    shape = a.shape
    strides = a.stride()

    utils.validate_idx(shape, start)
    utils.validate_exclusive_idx(shape, end)

    # Verifies end is strictly greater than start
    # (Collapse requires a non-empty interval)
    assert end > start

    length = 1
    stride = 1
    for idx in range(start, end):
        if idx != (end - 1):
            assert strides[idx] == strides[idx + 1] * shape[idx + 1]
        length = length * shape[idx]
        stride = stride * strides[idx]

    new_shape = shape[:start] + (length,) + shape[end:]
    new_strides = strides[:start] + (stride,) + shape[end:]

    return TensorMeta(a, shape=new_shape, strides=new_strides)


def _collapse_view_aten(a: Tensor, start: int, end: int) -> Tensor:
    # Short-circuits on null op
    if start == end - 1:
        return a

    dim_length = 1
    for idx in range(start, end):
        dim_length = dim_length * a.shape[idx]

    new_shape = a.shape[0:start] + (dim_length,) + a.shape[end:]

    return a.view(new_shape)


_collapse_view_doc = """
  Creates a view of a with the dimensions between
  start (inclusive) and end (exclusive) merged into a
  single dimension.

  If it's not possible to take such a view then an error
  is thrown. See collapse instead.

  The dimensions can be merged if and only if
  they are all "nested" with each other. That is, they all
  have the property that

  stride[i] = stride[i+1] * shape[i+1]

  for all i in [start, end - 1).
  """

collapse_view = _make_prim(
    meta=_collapse_view_meta,
    impl_aten=_collapse_view_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_collapse_view_doc,
)


def _split_dim_meta(a: TensorLikeType, dim: int, outer_length: int) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    utils.validate_idx(a.shape, dim)
    utils.validate_dim_length(outer_length)

    # Verifies the dim can be split with the specified lhs_length
    _inner_length = a.shape[dim] / outer_length
    inner_length: int = int(_inner_length)
    assert inner_length == _inner_length

    new_shape: List[int] = []
    new_strides: List[int] = []
    for idx in a.shape:
        if idx == dim:
            new_shape.extend((outer_length, inner_length))
            new_strides.extend((a.stride()[idx] * inner_length, a.stride()[idx]))
        else:
            new_shape.append(a.shape[idx])
            new_strides.append(a.stride()[idx])

    return TensorMeta(a, shape=new_shape, strides=new_strides)


def _split_dim_aten(a: Tensor, dim: int, outer_length: int) -> Tensor:
    inner_length = int(a.shape[dim] / outer_length)
    new_shape = a.shape[0:dim] + (outer_length, inner_length) + a.shape[dim + 1 :]

    return a.view(new_shape)


_split_dim_doc = """
  Creates a view of a with the given dimension (of length l) split
  into two dimensions, with the outer of the two having
  length outer_length and the inner of the two having computed
  length inner_length such outer_length * inner_length = l.
  """

# TODO: consider renaming split_dim_view
split_dim = _make_prim(
    meta=_split_dim_meta,
    impl_aten=_split_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_split_dim_doc,
)

# Note: allows dimensions to be specified redundantly
def _squeeze_meta(a: TensorLikeType, dimensions: Sequence) -> TensorLikeType:
    assert isinstance(a, TensorLike)

    for idx in dimensions:
        utils.validate_idx(a.shape, idx)
        assert a.shape[idx] == 1

    new_shape = []
    new_strides = []
    for idx in range(len(a.shape)):
        if idx in dimensions:
            continue

        new_shape.append(a.shape[idx])
        new_strides.append(a.stride()[idx])

    return TensorMeta(a, shape=new_shape, strides=new_strides)


def _squeeze_aten(a: Tensor, dimensions: Sequence) -> Tensor:
    for idx in dimensions:
        a = torch.squeeze(a, dim=idx)

    return a


_squeeze_doc = """
  Creates a view of the tensor with the specified dimensions removed.

  The removed dimensions must each have length one.
  """

squeeze = _make_prim(
    meta=_squeeze_meta,
    impl_aten=_squeeze_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_squeeze_doc,
)

#
# Shape operations
#
def collapse(a: Tensor, start: int, end: int) -> Tensor:
    """
    Wrapper around reshape that collapses a span of dimensions.

    See merge_dims for the corresponding view operation.
    """

    dim_length = 1
    for idx in range(start, end):
        dim_length = dim_length * a.shape[idx]

    new_shape = a.shape[0:start] + (dim_length,) + a.shape[end:]
    return reshape(a, new_shape)


# TODO: review stride logic
def _concatenate_meta(tensors: Sequence[TensorLikeType], dim: int) -> TensorLikeType:
    assert len(tensors) > 0

    for tensor in tensors:
        assert isinstance(tensor, TensorLike)

    utils.check_same_dtype(tensors)
    utils.check_same_device(tensors, allow_scalars=False)

    shape = tensors[0].shape
    utils.validate_idx(shape, dim)

    # Verifies same shape (except in the concat dimension)
    concat_length = 0
    for tensor in tensors:
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                concat_length = concat_length + length
            else:
                assert length == common_length

    new_shape = list(tensors[0].shape).copy()
    new_shape[dim] = concat_length
    return TensorMeta(
        tensors[0],
        shape=new_shape,
        strides=utils.make_contiguous_strides_for(new_shape),
    )


def _concatenate_aten(
    tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim: int
) -> Tensor:
    return torch.cat(tensors, dim)


_concatenate_doc = """
  Concatenates tensors along the specified dimension.

  The tensors' shapes must have the same rank and same length for other dimensions.
  """

concatenate = _make_prim(
    meta=_concatenate_meta,
    impl_aten=_concatenate_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_concatenate_doc,
)


# TODO: needs to return the proper meta tensor
def _reshape_meta(a: TensorLikeType, shape: Sequence):
    assert isinstance(a, TensorLike)
    utils.validate_shape(shape)

    # Validates the tensor and the requested shape have the
    # same number of elements
    numel = reduce(lambda acc, x: acc * x, shape)
    assert a.numel() == numel


def _reshape_aten(
    a: Tensor, shape: Union[torch.Size, List[int], Tuple[int, ...]]
) -> Tensor:
    return a.clone().reshape(shape).contiguous()


_reshape_doc = """
  Creates a contiguous tensor with the specified shape
  containing a copy of the data in a.
  """
reshape = _make_prim(
    meta=_reshape_meta,
    impl_aten=_reshape_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_reshape_doc,
)

#
# Conditional prims
#


def _select_meta(
    pred: TensorLikeType, a: TensorLikeType, b: TensorLikeType
) -> TensorLikeType:
    utils.check_same_device(pred, a, b, allow_scalars=True)
    utils.check_same_shape(pred, a, b)
    assert pred.dtype is torch.bool

    return _elementwise_meta(a, b)


def _select_aten(pred: Tensor, a: Tensor, b: Tensor) -> Tensor:
    return torch.where(pred, a, b)


_select_doc = """
  Selects elements from a and b according to pred.

  Where pred is true the result contains the element from a, and
  where pred is false the result contains the element from b.
  """

select = _make_prim(
    meta=_select_meta,
    impl_aten=_select_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_select_doc,
)

#
# Type conversions
#


def _convert_element_type_meta(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    # Type checks
    assert isinstance(a, TensorLike)
    assert isinstance(dtype, torch.dtype)

    return TensorMeta(a, dtype=dtype)


def _convert_element_type_aten(a: Tensor, dtype: torch.dtype) -> Tensor:
    return a.to(dtype)


_convert_element_type_doc = """
  Creates a copy of a tensor with the given dtype.
  """

convert_element_type = _make_prim(
    meta=_convert_element_type_meta,
    impl_aten=_convert_element_type_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_convert_element_type_doc,
)


def _device_put_meta(
    a: TensorLikeType, device: Union[str, torch.device]
) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    assert isinstance(device, (str, torch.device))

    return TensorMeta(a, device=utils.wrap_device(device))


def _device_put_aten(a: Tensor, device: Union[str, torch.device]) -> Tensor:
    return a.to(device)


_device_put_doc = """
  Creates a copy of a tensor on the given device.
  """

device_put = _make_prim(
    meta=_device_put_meta,
    impl_aten=_device_put_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_device_put_doc,
)

#
# Inplace operators
#


def _copy_to_meta(a: TensorLikeType, b: TensorLikeType):
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)

    # Validates the cast is safe
    a_typ = utils.dtype_to_type(a.dtype)
    b_typ = utils.dtype_to_type(b.dtype)
    if a_typ is not utils.get_higher_type(a_typ, b_typ):
        raise RuntimeError(str(b.dtype), " can't be cast safely to ", str(a.dtype), "!")

    # Validates the tensors have the same number of elements
    if a.numel() != b.numel():
        msg = "Attempting to copy {0} elements to a tensor with {1} elements!".format(
            b.numel(), a.numel()
        )
        raise RuntimeError(msg)

    return a


def _copy_to_aten(a: Tensor, b: Tensor) -> Tensor:
    return a.copy_(b)


_copy_to_doc = """
  Copies the data in b to a and returns the modified a.
  """

# TODO: Remove safe casting and implement on reference instead
copy_to = _make_prim(
    meta=_copy_to_meta,
    impl_aten=_copy_to_aten,
    return_type=RETURN_TYPE.INPLACE,
    doc=_copy_to_doc,
)


def _resize_meta(
    a: TensorLikeType, shape: Union[torch.Size, List[int], Tuple[int, ...]]
):
    assert a.numel() == 0
    return TensorMeta(a, shape=shape, strides=utils.make_contiguous_strides_for(shape))


def _resize_aten(a: Tensor, shape: ShapeType) -> Tensor:
    return a.resize_(shape)


_resize_doc = """
  Gives a tensor with no elements a new shape, returning the modified tensor.

  The tensor's strides are contiguous and its values are unitialized.
  """

# TODO: review support arbitrary resizes
resize = _make_prim(
    meta=_resize_meta,
    impl_aten=_resize_aten,
    return_type=RETURN_TYPE.INPLACE,
    doc=_resize_doc,
)


def _reduction_meta(inp, dims, *, output_dtype=None):
    """
    Meta function for single output reduction operations
    Stride logic is incorrect
    """
    assert isinstance(inp, TensorLike)
    if output_dtype is None:
        output_dtype = inp.dtype
    output_shape = utils.compute_reduction_output_shape(inp.shape, dims)
    return TensorMeta(shape=output_shape, dtype=output_dtype, device=inp.device)


def _bool_return_reduction_meta(inp, dims):
    return _reduction_meta(inp, dims, output_dtype=torch.bool)


_sum_doc = """
    Computes the sum of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_amax_doc = """
    Computes the maximum value of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_amin_doc = """
    Computes the minimum value of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """


sum = _make_prim(
    meta=_reduction_meta, impl_aten=torch.sum, return_type=RETURN_TYPE.NEW, doc=_sum_doc
)

prod = _make_prim(
    meta=_reduction_meta,
    impl_aten=torch.prod,
    return_type=RETURN_TYPE.NEW,
    doc=_sum_doc,
)

amax = _make_prim(
    meta=_reduction_meta,
    impl_aten=torch.amax,
    return_type=RETURN_TYPE.NEW,
    doc=_amax_doc,
)

amin = _make_prim(
    meta=_reduction_meta,
    impl_aten=torch.amin,
    return_type=RETURN_TYPE.NEW,
    doc=_amin_doc,
)

all = _make_prim(
    meta=_bool_return_reduction_meta,
    impl_aten=torch.all,
    return_type=RETURN_TYPE.NEW,
    doc="",
)

any = _make_prim(
    meta=_bool_return_reduction_meta,
    impl_aten=torch.any,
    return_type=RETURN_TYPE.NEW,
    doc="",
)
