from __future__ import annotations

from typing import Any, Union, Sequence, Optional, Tuple, List, Callable, Type
from enum import Enum
from functools import reduce, cmp_to_key
import operator
import weakref
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

import torch

# nvFuser imports are conditional on CUDA being available
if torch.cuda.is_available():
    from torch._C._nvfuser import DataType  # type: ignore[import]

    _torch_dtype_to_nvfuser_dtype_map = {
        torch.cdouble: DataType.ComplexDouble,
        torch.cfloat: DataType.ComplexFloat,
        torch.double: DataType.Double,
        torch.float: DataType.Float,
        torch.half: DataType.Half,
        torch.bfloat16: DataType.BFloat16,
        torch.long: DataType.Int,
        torch.int: DataType.Int32,
        torch.bool: DataType.Bool,
    }
else:
    _torch_dtype_to_nvfuser_dtype_map = {}


def getnvFuserDtype(dtype: torch.dtype):
    """
    Translates from torch.dtype to nvFuser's DataType enum
    """
    return _torch_dtype_to_nvfuser_dtype_map[dtype]


ShapeType = Union[torch.Size, List[int], Tuple[int, ...]]
StrideType = Union[List[int], Tuple[int, ...]]
DimsType = Union[int, List[int], Tuple[int, ...]]
DimsSequenceType = Union[List[int], Tuple[int, ...]]
NumberType = Union[bool, int, float, complex]
Number = (bool, int, float, complex)
DeviceLikeType = Union[str, torch.device]


torch_function_passthrough = {
    torch.Tensor.ndim.__get__,  # type: ignore[attr-defined]
    torch.Tensor.numel,
    torch.Tensor.stride,
    torch.Tensor.dtype.__get__,  # type: ignore[attr-defined]
    torch.Tensor.shape.__get__,  # type: ignore[attr-defined]
    torch.Tensor.device.__get__,  # type: ignore[attr-defined]
    # For TorchRefsMode only
    torch.Tensor.__format__,
    torch.Tensor.__repr__,
}


TensorLikeType = torch.Tensor
TensorLike = torch.Tensor
TensorSequenceType = Union[List[TensorLikeType], Tuple[TensorLikeType, ...]]
TensorOrNumberLikeType = Union[TensorLikeType, NumberType]


# In order to keep things like aliasing relationships and storage
# consistent wrt/meta tensors, FakeTensors own a FakeTensorMode
# which caches conversions to Meta Tensors. We would like to use
# one consistent mode among along FakeTensors, which we store here.
# We store a weakref, so that when all previous FakeTensors are
# the present mode will also deallocate. FakeTensorMode holds onto
# tensors that are converted to Meta so we don't want to persist it
# longer than necessary.x
prim_fake_mode_ref = None


def get_prim_fake_mode():
    global prim_fake_mode_ref
    if prim_fake_mode_ref is None or prim_fake_mode_ref() is None:
        mode = FakeTensorMode()
        prim_fake_mode_ref = weakref.ref(mode)
        return mode
    else:
        return prim_fake_mode_ref()


def TensorMeta(
    tensorlike: Optional[Union[NumberType, torch.Tensor]] = None,
    *,
    shape: Optional[ShapeType] = None,
    strides: Optional[StrideType] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
):
    if isinstance(tensorlike, Number):
        assert not shape and (shape is None or isinstance(shape, Sequence))
        assert not strides and (strides is None or isinstance(strides, Sequence))
        inferred_shape: Tuple[int, ...] = ()
        inferred_strides: Tuple[int, ...] = ()
        inferred_dtype = type_to_dtype(type(tensorlike))
        inferred_device = torch.device("cpu")
        # TODO: This looks wrong, a number that is wrapped into a tensor
        # needs to behave differently than a scalar tensor for type
        # promotion purposes
    elif tensorlike is not None:
        assert isinstance(tensorlike, torch.Tensor)
        inferred_shape = tuple(tensorlike.shape)
        inferred_strides = tuple(tensorlike.stride())
        inferred_dtype = tensorlike.dtype
        inferred_device = tensorlike.device
    else:
        # If no tensorlike "example" is given then all metadata
        # must be provided explicitly
        assert shape is not None
        assert strides is not None
        assert dtype is not None
        assert device is not None

    shape = inferred_shape if shape is None else tuple(shape)
    strides = inferred_strides if strides is None else tuple(strides)
    dtype = inferred_dtype if dtype is None else dtype
    device = inferred_device if device is None else device

    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(tensorlike, FakeTensor):
        mode = tensorlike.fake_mode
    else:
        mode = get_prim_fake_mode()

    if device.type == "meta":
        return torch.empty_strided(shape, strides, dtype=dtype, device="meta")
    else:
        return FakeTensor(
            mode,
            torch.empty_strided(shape, strides, dtype=dtype, device="meta"),
            device,
        )


def same_shape(a: ShapeType, b: ShapeType) -> bool:
    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if x != y:
            return False

    return True


# TODO: look at using torch.testing.assert_close instead with an option
#   to just compare metadata
def compare_tensor_meta(a: TensorLikeType, b: TensorLikeType):
    """
    Checks that two tensor likes have the same shape,
    dtype and device.

    In the future this will validate additional metadata, like
    strides.
    """
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)

    if not same_shape(a.shape, b.shape):
        msg = "Shapes {0} and {1} are not equal!".format(a.shape, b.shape)
        raise AssertionError(msg)

    if a.dtype != b.dtype:
        msg = "Dtypes {0} and {1} are not equal!".format(a.dtype, b.dtype)
        raise AssertionError(msg)

    if a.device != b.device:
        # Handles special cuda:0 vs cuda case
        # TODO: we should review why this happens and see about fixing it
        if (str(a.device) == "cuda:0" or str(a.device) == "cuda") and (
            str(b.device) == "cuda:0" or str(b.device) == "cuda"
        ):
            pass
        else:
            msg = "Devices {0} and {1} are not equal!".format(a.device, b.device)
            raise AssertionError(msg)

    # Stride checking is currently disabled, see https://github.com/pytorch/pytorch/issues/78050
    # same_strides, idx = check_significant_strides(a, b)
    # if not same_strides:
    #     msg = "Stride mismatch! Strides are {0} and {1} (mismatched at {2})!".format(
    #         a.stride(), b.stride(), idx
    #     )
    # raise RuntimeError(msg)


def check_significant_strides(
    a: TensorLikeType, b: TensorLikeType
) -> Tuple[bool, Optional[int]]:
    # NOTE: only on CUDA because CPU elementwise strides are incorrect in PyTorch
    # See https://github.com/pytorch/pytorch/issues/77553
    # Only compares strides that are "meaningful" -- strides for dimensions with length > 1
    # and for tensors with more than one element
    if (a.device.type == "cuda" or b.device.type == "cuda") and a.numel() > 0:
        for idx in range(a.ndim):
            if a.stride()[idx] != b.stride()[idx] and a.shape[idx] > 1:
                return False, idx

    return True, None


def is_contiguous(a: TensorLikeType) -> bool:
    """
    Tests whether a tensor is contiguous or not.

    Tensors are contiguous when they have no elements,
    or when they have "nested" strides.
    """
    if a.numel() == 0:
        return True

    expected_stride = 1
    for x, y in reversed(tuple(zip(a.shape, a.stride()))):
        # Skips checking strides when a dimension has length 1
        if x == 1:
            continue

        if y != expected_stride:
            return False
        expected_stride = expected_stride * x

    return True


# NOTE: Based on the implementation in TensorIterator.cpp, but note that
# the note [Computing output strides] is incorrect, because it
# says that strides will be preserved even if they are not
# "non overlapping and dense", but this is incorrect. The
# output of elementwise operations are always given
# non overlapping and dense strides.
# This is also INCORRECT because it does not model TensorIterator's
# short-circuit, which can cause different strides.
def compute_elementwise_output_strides(*tensors) -> Tuple[int, ...]:
    """
    Computes the output strides for elementwise operations.
    """

    if len(tensors) == 0:
        msg = "Can't compute elementwise output strides for zero tensors!"
        raise ValueError(msg)

    check_same_shape(*tensors, allow_cpu_scalar_tensors=True)

    # Filters the tensors to actual tensors
    tensors = tuple(
        a for a in tensors if isinstance(a, TensorLike) and not is_cpu_scalar_tensor(a)
    )

    # Short-circuits for CPU scalar case
    if len(tensors) == 0:
        return ()

    # Short-circuits for shapes with zero or one dimensions
    # TODO: are these necessary?
    ndim = tensors[0].ndim
    if ndim == 0:
        return ()
    if ndim == 1:
        return (1,)

    shape = tensors[0].shape

    def _cmp(idx_a, idx_b):
        for tensor in tensors:
            stride_a = tensor.stride()[idx_a]
            stride_b = tensor.stride()[idx_b]

            if stride_a == 0 or stride_b == 0:
                continue

            if stride_a < stride_b:
                return -1

            if stride_a > stride_b:
                return 1

            # stride_a == stride_b
            if shape[idx_a] > shape[idx_b]:
                return 1

            # NOTE: this case is missing in the C++ impl
            if shape[idx_a] < shape[idx_b]:
                return -1

        # Note: this case is hit if all strides are zero,
        # or all strides are equal and all dimensions have the same length
        return 0

    perm = tuple(range(ndim))
    perm = tuple(sorted(perm, key=cmp_to_key(_cmp), reverse=True))

    permuted_shape = [-1] * ndim
    for idx, x in enumerate(perm):
        permuted_shape[idx] = shape[x]

    new_strides = make_contiguous_strides_for(permuted_shape)
    permuted_strides = [-1] * ndim
    for idx, x in enumerate(perm):
        permuted_strides[x] = new_strides[idx]

    return tuple(permuted_strides)


#
# Common helper functions
#


def validate_dim_length(length: int):
    """
    Validates that an object represents a valid
    dimension length.
    """

    assert isinstance(length, int)
    assert length >= 0


def validate_shape(shape: ShapeType):
    """
    Validates that a sequence represents a valid shape.
    """

    assert isinstance(shape, Sequence)
    for l in shape:
        validate_dim_length(l)


def validate_strides(strides: StrideType):
    """
    Verifies the object specifies valid strides.
    """

    assert isinstance(strides, Sequence)
    for stride in strides:
        assert stride >= 0


def validate_idx(rank: int, idx: int):
    """
    Validates that idx is a valid index for the given shape.
    Assumes the index is already canonicalized.
    """

    assert isinstance(idx, int)
    assert isinstance(rank, int)

    assert idx >= 0 and idx < rank or idx == 0


def validate_dimension_indices(rank: int, indices: DimsSequenceType):
    for idx in indices:
        validate_idx(rank, idx)


def validate_exclusive_idx(rank: int, ex_idx: int):
    """
    Validates that ex_idx is a valid exclusive index
    for the given shape.
    """

    assert isinstance(ex_idx, int)
    assert isinstance(rank, int)
    assert ex_idx > 0 and ex_idx <= rank


# "Wraps" a dim (up to one time) for the given rank, allowing
# dims to be specified using negative indices
def canonicalize_dim(rank: int, idx: int) -> int:
    # TODO: add a comment for why this is
    _rank = rank if rank != 0 else 1

    if idx >= 0 and idx < _rank:
        return idx

    if idx < 0:
        _idx = idx + _rank
    else:
        _idx = idx

    if _idx < 0 or _idx > _rank:
        # Same error message as in aten/src/ATen/WrapDimUtils.h:49
        msg = "Dimension out of range (expected to be in range of [{0}, {1}], but got {2})".format(
            -rank, rank - 1, idx
        )
        raise IndexError(msg)

    return _idx


# Takes a dimension or sequence of dimensions and "wraps" them,
# mapping negative offsets to positive ones
def canonicalize_dims(rank: int, indices: DimsType) -> DimsType:
    if isinstance(indices, int):
        return canonicalize_dim(rank, indices)

    return tuple(canonicalize_dim(rank, x) for x in indices)


def is_valid_permutation(rank: int, perm: DimsSequenceType) -> bool:
    """
    Validates that perm is a permutation of length rank.
    """

    if not isinstance(perm, Sequence):
        return False

    if not (tuple(sorted(perm)) == tuple(range(0, rank))):
        return False

    return True


def is_same_shape(a: Sequence, b: Sequence) -> bool:
    """
    Compares two shapes a and b, returning True if they are the same
    (their ranks and corresponding lengths match) and False otherwise.
    """

    return tuple(a) == tuple(b)


def is_cpu_scalar_tensor(a: Any) -> bool:
    return isinstance(a, TensorLike) and a.ndim == 0 and a.device.type == "cpu"


def check_same_device(*args, allow_cpu_scalar_tensors):
    """
    Checks that all Tensors in args have the same device.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensor objects in args have different devices, unless one is a CPU scalar tensor and allow_cpu_scalar_tensors is True
    """
    # Short-circuits if all (one or fewer) arguments are trivially on the same device
    if len(args) <= 1:
        return

    # Note: cannot initialize device to the first arg's device (it may not have one)
    device = None
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                continue

            if device is None:
                device = arg.device

            if device != arg.device:
                msg = (
                    "Tensor on device "
                    + str(arg.device)
                    + " is not on the expected device "
                    + str(device)
                    + "!"
                )
                raise RuntimeError(msg)
        else:
            msg = (
                "Unexpected type when checking for same device, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)


def canonicalize_device(device: DeviceLikeType) -> torch.device:
    if isinstance(device, torch.device):
        return device

    assert isinstance(device, str)
    return torch.device(device)


# Asserts if any of the following are true:
#   - a non-scalar or non-Tensor is given
#   - the shape of any tensors is distinct
def check_same_shape(*args, allow_cpu_scalar_tensors: bool):
    """
    Checks that all Tensors in args have the same shape.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensor objects in args have different devices
    """
    shape = None

    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                continue

            if shape is None:
                shape = arg.shape

            if not is_same_shape(shape, arg.shape):
                msg = "Shape {0} is not the expected shape {1}!".format(
                    arg.shape, shape
                )
                raise RuntimeError(msg)
        else:
            msg = (
                "Unexpected type when checking for same shape, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)


def extract_shape(*args, allow_cpu_scalar_tensors: bool) -> Optional[ShapeType]:
    shape = None
    scalar_shape = None

    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                scalar_shape = arg.shape
                continue

            if shape is None:
                shape = arg.shape

            if not is_same_shape(shape, arg.shape):
                return None
        else:
            return None

    return shape if shape is not None else scalar_shape


def extract_shape_from_varargs(
    shape: Union[ShapeType, Tuple[ShapeType]]
) -> Tuple[int, ...]:
    """
    Returns a shape from varargs.

    In PyTorch, operations that accept shapes often accept them as varargs, like
    foo(*shape). However a user can pass the shape as a sequence of integers,
    like this:

      foo(1, 2, 3)

    or as a sequence of integers

      foo((1, 2, 3))

    In the first case shape will be a tuple of integers, and in the second case it's a tuple
    containing a tuple of integers. This validates those inputs and canonicalizes them
    to a tuple of integers.
    """

    # Handles tuple unwrapping
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]

    validate_shape(shape)  # type: ignore[arg-type]
    return shape  # type: ignore[return-value]


_integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_low_precision_dtypes = (torch.float16, torch.bfloat16, torch.complex32)
_float_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_complex_dtypes = (torch.complex32, torch.complex64, torch.complex128)


def is_boolean_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype is torch.bool


def is_integer_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype in _integer_dtypes


def is_low_precision_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype in _low_precision_dtypes


def is_float_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype in _float_dtypes


def is_complex_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype in _complex_dtypes


def is_grad_dtype(dtype: torch.dtype) -> bool:
    """
    Checks if the dtype can require a gradient.
    """
    return is_float_dtype(dtype) or is_complex_dtype(dtype)


_complex_to_real_dtype_map = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
}

_real_to_complex_dtype_map = {
    torch.float16: torch.complex32,
    torch.bfloat16: torch.complex64,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def corresponding_real_dtype(dtype: torch.dtype) -> torch.dtype:
    return _complex_to_real_dtype_map[dtype]


def corresponding_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return _real_to_complex_dtype_map[dtype]


def dtype_to_type(dtype: torch.dtype) -> type:
    """
    Computes the corresponding Python type (AKA "type kind") for the
    given dtype.
    """
    assert isinstance(dtype, torch.dtype)

    if dtype is torch.bool:
        return bool
    if dtype in _integer_dtypes:
        return int
    if dtype in _float_dtypes:
        return float
    if dtype in _complex_dtypes:
        return complex

    raise ValueError("Invalid dtype!")


def type_to_dtype(typ: type) -> torch.dtype:
    """
    Computes the corresponding dtype for a Number type.
    """

    assert isinstance(typ, type)

    if typ is bool:
        return torch.bool
    if typ is int:
        return torch.long
    if typ is float:
        return torch.get_default_dtype()
    if typ is complex:
        return corresponding_complex_dtype(torch.get_default_dtype())

    raise ValueError("Invalid type!")


_ordered_types = (bool, int, float, complex)


def get_higher_type(a: type, b: type) -> type:
    """
    Returns the higher of the two given Number types.

    The types are ordered bool -> int -> float -> complex.
    """
    # Type checking
    assert a in _ordered_types
    assert b in _ordered_types

    if a is b:
        return a

    for typ in _ordered_types:
        if a is typ:
            return b
        if b is typ:
            return a

    raise ValueError("Unknown Python scalar type!")


# Returns the higher of two torch datatypes a and b or, if the two
#   are not ordered relative to each other, the next
#   higher datatype
def get_higher_dtype(
    a: Optional[Union[torch.dtype, TensorLikeType, NumberType]],
    b: Optional[Union[torch.dtype, TensorLikeType, NumberType]],
) -> Optional[torch.dtype]:
    """
    Computes the "lowest" datatype that is weakly
    "higher" than both a and b.
    """

    # Type checking
    assert a is None or isinstance(a, (torch.dtype, TensorLike, Number))
    assert b is None or isinstance(b, (torch.dtype, TensorLike, Number))

    def _extract_dtype(
        x: Optional[Union[torch.dtype, TensorLikeType, NumberType]]
    ) -> Optional[torch.dtype]:
        if x is None:
            return None
        if isinstance(x, torch.dtype):
            return x
        if isinstance(x, TensorLike):
            return x.dtype
        if isinstance(x, Number):
            return type_to_dtype(type(x))

        raise RuntimeError("Unexpected type given to _extract_dtype!")

    a, b = _extract_dtype(a), _extract_dtype(b)

    if a is b:
        return a

    if a is None:
        return b

    if b is None:
        return a

    ordered_datatypes = (
        (torch.bool,),
        (torch.uint8, torch.int8),
        (torch.int16,),
        (torch.int32,),
        (torch.int64,),
        (torch.float16, torch.bfloat16),
        (torch.float32,),
        (torch.float64,),
        (torch.complex32,),
        (torch.complex64,),
        (torch.complex128,),
    )

    for idx, dtypes in enumerate(ordered_datatypes):
        if a in dtypes and b in dtypes:
            return ordered_datatypes[idx + 1][0]
        if a in dtypes:
            return b
        if b in dtypes:
            return a

    raise RuntimeError("Unexpected termination!")


# TODO: maybe unify with can_cast_to?
def is_weakly_lesser_type(a: type, b: type) -> bool:
    """
    Compares two types, a and b, returning True if a is weakly "less" than b.

    The comparison is determined by the following type ordering: bool, int, float, complex.
    """
    ordered_types = (
        bool,
        int,
        float,
        complex,
    )

    assert a in ordered_types
    assert b in ordered_types

    for typ in ordered_types:
        if a == typ:
            return True
        if b == typ:
            return False

    raise RuntimeError("Unexpected termination!")


def can_safe_cast_to(*, cast_to: torch.dtype, cast_from: torch.dtype) -> bool:
    for fn in (is_complex_dtype, is_float_dtype, is_integer_dtype, is_boolean_dtype):
        if fn(cast_to):
            return True
        if fn(cast_from):
            return False

    raise ValueError("Received unknown dtypes {0}, {1}!".format(cast_to, cast_from))


def check_same_dtype(*args):
    """
    Checks that all Tensors in args have the same device and that all Numbers have the
    same corresponding Python type.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensors objects in args have different dtypes
      - two Number objects in args have different types
      - there are Tensors and Numbers in args, and one of those Tensors corresponding
          Python types is different from the type of one of those Numbers
    """
    full_dtype = None
    scalar_type = None

    for arg in args:
        if isinstance(arg, Number):
            # Scalar type checking is disabled (and may be removed in the future)
            continue
            # if scalar_type is None:
            #     scalar_type = type(arg)

            # if scalar_type is not type(arg):
            #     msg = (
            #         "Scalar of type "
            #         + str(type(arg))
            #         + " is not the expected type of "
            #         + str(scalar_type)
            #         + "!"
            #     )
            #     raise RuntimeError(msg)
        elif isinstance(arg, TensorLike):
            if full_dtype is None:
                full_dtype = arg.dtype
            if scalar_type is None:
                scalar_type = dtype_to_type(arg.dtype)

            if full_dtype is not arg.dtype:
                msg = (
                    "Tensor with dtype "
                    + str(arg.dtype)
                    + " is not the expected dtype of "
                    + str(full_dtype)
                    + "!"
                )
                raise RuntimeError(msg)

            arg_type = dtype_to_type(arg.dtype)
            if arg_type is not scalar_type:
                msg = (
                    "Tensor with corresponding Python type "
                    + str(arg_type)
                    + " is not the expected type of "
                    + str(scalar_type)
                    + "!"
                )
                raise RuntimeError(msg)
        else:
            msg = (
                "Unexpected type when checking for same dtype, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)


# Maps datatypes to their computation types for elementwise operations
_computation_dtype_map = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
    torch.complex32: torch.complex64,
}


def get_computation_dtype(dtype: torch.dtype) -> torch.dtype:
    return _computation_dtype_map.get(dtype, dtype)


class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    NO_OPMATH = (1,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)
    BOOL_TO_LONG = (5,)


class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME = (0,)
    COMPLEX_TO_FLOAT = (1,)  # for complex types outputs corresponding real type
    KEEP_PROMOTED_TYPE = (2,)  # keep output in opmath type, needed for mean
    ALWAYS_BOOL = (3,)


# TODO: document type promotion kinds
def elementwise_dtypes(
    *_args,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
) -> Tuple[torch.dtype, torch.dtype]:
    """
    Computes the computation and result dtypes for elementwise type promotion
    on the given arguments and with the given elementwise type promotion kind.

    Note that not all inputs to an elementwise operation necessarily participate in type promotion.
    For example, the "alpha" parameter of torch.add does not participate in type promotion,
    although it may be cast to the Python type corresponding to the computation dtype that
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

    The DEFAULT type promotion kind computes per above, and then uses the result dtype to pick a computation
    dtype by mapping low precision floating point and complex dtypes as follows:

      float16   -> float32
      bfloat16  -> float32
      complex32 -> complex64

    This is referred to as "op math", and the NO_OPMATH type promotion kind disables this mapping, making the
    computation dtype the same as the result dtype when it's selected. NO_OPMATH is appropriate for kernels
    which perform no mathematical operations on their tensors (see below for examples).

    The INT_TO_FLOAT type promotion kind maps boolean and integer maps result dtypes to the default floating point dtype,
    and computation dtypes to the appropriate op math dtype.

    The COMPLEX_TO_FLOAT type promotion kind maps complex result dtypes to the corresponding float dtype, following this
    mapping:

        complex32  -> float16
        complex64  -> float32
        complex128 -> float64

    Note that COMPLEX_TO_FLOAT derives the computation dtype as the DEFAULT setting does.

    The BOOL_TO_LONG type promotion kind maps boolean computation and result dtypes to long.

    The ALWAYS_BOOL type promotion kind always sets the result dtype to bool.

    Example operators for each type promotion option:
      DEFAULT                 : add
      NO_OPMATH               : where, nextafter, cat
      INT_TO_FLOAT            : sin
      COMPLEX_TO_FLOAT        : abs
      BOOL_TO_LONG            : pow
      ALWAYS_BOOL             : eq

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
            highest_type = get_higher_type(highest_type, type(x))
        else:
            # x is a TensorLike
            highest_type = get_higher_type(highest_type, dtype_to_type(x.dtype))

    result_dtype = None

    def _find_highest_dtype_filtered(
        args, filter, *, float_as_complex=False
    ) -> Optional[torch.dtype]:
        zero_dim_tensor_dtype = None
        one_plus_dim_tensor_dtype = None
        for x in args:
            if isinstance(x, TensorLike) and filter(x.dtype):
                _dtype = x.dtype
                if float_as_complex and is_float_dtype(_dtype):
                    _dtype = corresponding_complex_dtype(_dtype)
                if x.ndim == 0:
                    zero_dim_tensor_dtype = get_higher_dtype(
                        zero_dim_tensor_dtype, _dtype
                    )
                else:
                    # x.ndim > 0
                    one_plus_dim_tensor_dtype = get_higher_dtype(
                        one_plus_dim_tensor_dtype, _dtype
                    )

        # Prefers dtype of tensors with one or more dimensions
        if one_plus_dim_tensor_dtype is not None:
            return one_plus_dim_tensor_dtype

        return zero_dim_tensor_dtype

    if highest_type is float:
        result_dtype = _find_highest_dtype_filtered(args, is_float_dtype)
        result_dtype = (
            torch.get_default_dtype() if result_dtype is None else result_dtype
        )
    elif highest_type is complex:
        result_dtype = _find_highest_dtype_filtered(
            args,
            lambda x: is_float_dtype(x) or is_complex_dtype(x),
            float_as_complex=True,
        )
        if result_dtype is None:
            result_dtype = corresponding_complex_dtype(torch.get_default_dtype())
    elif highest_type is int:
        result_dtype = _find_highest_dtype_filtered(args, is_integer_dtype)
        result_dtype = torch.long if result_dtype is None else result_dtype
    else:
        # highest_type is bool
        result_dtype = torch.bool

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT:
        return get_computation_dtype(result_dtype), result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH:
        return result_dtype, result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
        if is_integer_dtype(result_dtype) or is_boolean_dtype(result_dtype):
            result_dtype = torch.get_default_dtype()
        return get_computation_dtype(result_dtype), result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
        # NOTE: computation can still occur in a complex dtype
        computation_dtype = get_computation_dtype(result_dtype)
        if is_complex_dtype(result_dtype):
            result_dtype = corresponding_real_dtype(result_dtype)
        return computation_dtype, result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG:
        if is_boolean_dtype(result_dtype):
            return torch.long, torch.long
        return get_computation_dtype(result_dtype), result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        return get_computation_dtype(result_dtype), torch.bool
    else:
        raise ValueError(
            "Unknown type promotion kind {0}".format(str(type_promotion_kind))
        )


def reduction_dtypes(
    arg,
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.dtype, Optional[torch.dtype]]:
    # even though some reductions, like amin or amax, don't strictly require type promotion,
    # all the math ops (including comparisons) are still defined only for a computation type,
    # so promotion will still happen. We are doing it explicitly here
    inp_dtype = dtype if dtype is not None else arg.dtype
    computation_dtype = get_computation_dtype(inp_dtype)
    if (
        output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.SAME
        or output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    ):
        result_dtype = dtype if dtype else arg.dtype
        if (
            output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
            and is_complex_dtype(result_dtype)
        ):
            result_dtype = corresponding_real_dtype(result_dtype)
    elif output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE:
        result_dtype = None
    else:  # ALWAYS_BOOL
        result_dtype = torch.bool
    return computation_dtype, result_dtype


def make_contiguous_strides_for(shape: ShapeType) -> Tuple[int, ...]:
    validate_shape(shape)
    if not shape:
        return ()

    multiplier = 1
    strides = []
    for l in reversed(shape):
        if l != 0:
            strides.append(multiplier)
            multiplier = l * multiplier
        else:
            strides.append(multiplier)

    result = tuple(reversed(strides))
    return result


def compute_reduction_output_shape(
    shape: ShapeType, dimensions: Sequence
) -> Tuple[int, ...]:
    for idx in dimensions:
        validate_idx(len(shape), idx)

    new_shape = []
    for idx in range(len(shape)):
        if idx in dimensions:
            continue

        new_shape.append(shape[idx])

    return tuple(new_shape)


def validate_no_repeating_dims(dims: Sequence):
    if len(dims) != len(set(dims)):
        raise RuntimeError("duplicate value in the list of dims")


def reduction_dims(shape: ShapeType, dims: Optional[Sequence]) -> Tuple[int, ...]:
    if dims is None:
        return tuple(range(len(shape)))
    dims = tuple(canonicalize_dim(len(shape), idx) for idx in dims)
    validate_no_repeating_dims(dims)
    return dims


def check_in_bounds_for_storage(
    a: torch._TypedStorage, shape: ShapeType, strides: StrideType, storage_offset: int
):
    """
    Determines if the given shape, strides, and offset are valid for the given storage.
    """

    # Short-circuits if the shape has no elements
    if reduce(operator.mul, shape) == 0:
        return

    length = a.size() - storage_offset
    max_offset = 0
    for x, y in zip(shape, strides):
        max_offset = max_offset + (x - 1) * y

    if max_offset >= length:
        required_length = max_offset + storage_offset
        msg = (
            "Can't view a storage of size {0} with an offset of {1}, shape of {2}, and strides of {3}, "
            "which requires a storage of size {4}".format(
                a.size(), storage_offset, str(shape), str(strides), required_length
            )
        )
        raise ValueError(msg)


def check(
    b: bool, s: Callable[[], str], exc_type: Type[Exception] = RuntimeError
) -> None:
    """
    Helper function for raising an error_type (default: RuntimeError) if a boolean condition fails.
    Error message is a callable producing a string (to avoid wasting time
    string formatting in non-error case, and also to make it easier for torchdynamo
    to trace.)
    """
    if not b:
        raise exc_type(s())
