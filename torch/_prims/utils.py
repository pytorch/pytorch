from __future__ import annotations

from numbers import Number
from typing import Any, Union, Sequence, Optional, Callable, Dict, Tuple, List
from functools import reduce

import torch

ShapeType = Union[torch.Size, List[int], Tuple[int, ...]]
StrideType = Union[List[int], Tuple[int, ...]]
DimsType = Union[int, List[int], Tuple[int, ...]]


class TensorMeta(object):
    """
    Temporary helper class to model tensor metadata.

    To be replaced with an actual meta tensor.
    """

    def __init__(
        self,
        tensorlike: Optional[Union[TensorMeta, Number, torch.Tensor]] = None,
        *,
        shape: Optional[ShapeType] = None,
        strides: Optional[StrideType] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):

        self.shape: Tuple[int, ...]
        self.strides: Tuple[int, ...]
        self.dtype: torch.dtype
        self.device: torch.device

        if isinstance(tensorlike, Number):
            assert not shape and (shape is None or isinstance(shape, Sequence))
            assert not strides and (strides is None or isinstance(strides, Sequence))
            self.shape = ()
            self.strides = ()
            self.dtype = type_to_dtype(type(tensorlike))
            self.device = torch.device("cpu")
        elif tensorlike is not None:
            assert isinstance(tensorlike, (TensorMeta, torch.Tensor))
            self.shape = tuple(tensorlike.shape)
            self.strides = tuple(tensorlike.stride())
            self.dtype = tensorlike.dtype
            self.device = tensorlike.device
        else:
            # If no tensorlike "example" is given then all metadata
            # must be provided explicitly
            assert shape is not None
            assert strides is not None
            assert dtype is not None
            assert device is not None

        # Sets metadata from kwargs, possibly overriding metadata from
        # the example tensorlike
        self.shape = self.shape if shape is None else tuple(shape)
        self.strides = self.strides if strides is None else tuple(strides)
        self.dtype = self.dtype if dtype is None else dtype
        self.device = self.device if device is None else device

        # Computes derived properties
        self.ndim = len(self.shape)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Optional[Dict] = None,
    ):
        if kwargs is None:
            kwargs = {}

        if not hasattr(func, "meta"):
            raise ValueError("Callable {0} has no meta function!".format(func.__name__))

        return func.meta(*args, **kwargs)  # type: ignore[attr-defined]

    def __repr__(self):
        return f"TensorMeta(dtype={self.dtype}, device={self.device}, shape={self.shape}, strides={self.strides})"

    def stride(self):
        return self.strides

    def numel(self):
        if len(self.shape) == 0:
            return 1

        return reduce(lambda x, acc: x * acc, self.shape, 1)


TensorLikeType = Union[torch.Tensor, TensorMeta]
TensorLike = (torch.Tensor, TensorMeta)


# TODO: look at using torch.testing.assert_close instead with an option
#   to just compare metadata
def compare_tensor_meta(a: TensorLikeType, b: TensorLikeType):
    """
    Checks that two tensor likes have the same shape,
    dtype, and device.

    In the future this will validate additional metadata, like
    strides.
    """
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)

    for x, y in zip(a.shape, b.shape):
        if x != y:
            msg = "Shapes {0} and {1} are not equal!".format(a.shape, b.shape)
            raise AssertionError(msg)

    if a.dtype != b.dtype:
        msg = "Dtypes {0} and {1} are not equal!".format(a.dtype, b.dtype)
        raise AssertionError(msg)

    if a.device != b.device:
        msg = "Devices {0} and {1} are not equal!".format(a.device, b.device)
        raise AssertionError(msg)


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


def validate_shape(shape: Sequence):
    """
    Validates that a sequence represents a valid shape.
    """

    assert isinstance(shape, Sequence)
    for l in shape:
        validate_dim_length(l)


def validate_idx(shape: Sequence, idx: int):
    """
    Validates that idx is a valid idx for the given shape.
    """

    assert isinstance(idx, int)
    assert idx >= 0 and idx < len(shape)


def validate_exclusive_idx(shape: Sequence, ex_idx: int):
    """
    Validates that ex_idx is a valid exclusive index
    for the given shape.
    """

    assert isinstance(ex_idx, int)
    assert ex_idx > 0 and ex_idx <= len(shape)


def canonicalize_idx(shape: Sequence, idx: int):
    validate_idx(shape, idx)
    if idx < 0:
        idx = idx + len(shape)
    return idx


def validate_permutation(rank: int, perm: Sequence):
    """
    Validates that perm is a permutation of length rank.
    """

    assert isinstance(perm, Sequence)
    assert tuple(sorted(perm)) == tuple(range(0, rank))


def is_same_shape(a: Sequence, b: Sequence):
    """
    Compares two shapes a and b, returning True if they are the same
    (their ranks and corresponding lengths match) and False otherwise.
    """

    return tuple(a) == tuple(b)


def check_same_device(*args, allow_scalars):
    """
    Checks that all Tensors in args have the same device.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - args contains an object whose type is Number and allow_scalar is False
      - two Tensor objects in args have different devices
    """
    # Short-circuits if all (one or fewer) arguments are trivially on the same device
    if len(args) <= 1:
        return

    # Note: cannot initialize device to the first arg's device (it may not have one)
    device = None
    for arg in args:
        if isinstance(arg, Number):
            if not allow_scalars:
                msg = "Found a scalar when checking for same device but scalars not allowed!"
                raise RuntimeError(msg)
        elif isinstance(arg, TensorLike):
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


# Asserts if any of the following are true:
#   - a non-scalar or non-Tensor is given
#   - the shape of any tensors is distinct
def check_same_shape(*args):
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


_integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_float_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_complex_dtypes = (torch.complex64, torch.complex128)


def is_boolean_dtype(dtype: torch.dtype) -> bool:
    return dtype is torch.bool


def is_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in _integer_dtypes


def is_float_dtype(dtype: torch.dtype) -> bool:
    return dtype in _float_dtypes


def is_complex_dtype(dtype: torch.dtype) -> bool:
    return dtype in _complex_dtypes


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


_type_to_dtype_map = {
    bool: torch.bool,
    int: torch.int64,
    float: torch.float64,
    complex: torch.complex128,
}


def type_to_dtype(typ: type) -> torch.dtype:
    """
    Computes the corresponding dtype for a Number type.
    """
    return _type_to_dtype_map[typ]


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
    a: Union[torch.dtype, torch.Tensor, Number],
    b: Union[torch.dtype, torch.Tensor, Number],
) -> torch.dtype:
    """
    Computes the "lowest" datatype that is weakly
    "higher" than both a and b.
    """

    # Type checking
    assert isinstance(a, (torch.dtype, torch.Tensor, Number))
    assert isinstance(b, (torch.dtype, torch.Tensor, Number))

    def _extract_dtype(x: Union[torch.dtype, torch.Tensor, Number]) -> torch.dtype:
        if isinstance(x, torch.dtype):
            return x
        if isinstance(x, torch.Tensor):
            return x.dtype
        if isinstance(x, Number):
            return type_to_dtype(type(x))

        raise RuntimeError("Unexpected type given to _extract_dtype!")

    a, b = _extract_dtype(a), _extract_dtype(b)

    if a is b:
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


def is_lesser_type(a: type, b: type) -> bool:
    """
    Compares two types, a and b, returning True if a is "less" than b.

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
            if scalar_type is None:
                scalar_type = type(arg)

            if scalar_type is not type(arg):
                msg = (
                    "Scalar of type "
                    + str(type(arg))
                    + " is not the expected type of "
                    + str(scalar_type)
                    + "!"
                )
                raise RuntimeError(msg)
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


def wrap_scalar(a: Number) -> torch.Tensor:
    """
    Wraps a Number into a Tensor of corresponding dtype.
    """
    return torch.tensor(a, dtype=type_to_dtype(type(a)))


def wrap_scalars(*args):
    """
    Wraps all Numbers in args using wrap_scalar.
    """

    def _maybe_wrap_scalar(x):
        if isinstance(x, Number):
            return wrap_scalar(x)
        return x

    return (_maybe_wrap_scalar(x) for x in args)


def wrap_device(d: Union[str, torch.device]) -> torch.device:
    """
    Wraps strings into torch.device objects.

    Given torch.device objects are returned unmodified.
    """

    assert isinstance(d, (str, torch.device))
    if isinstance(d, str):
        return torch.device(d)

    return d


def make_contiguous_strides_for(shape: Sequence) -> Tuple[int, ...]:
    validate_shape(shape)
    if not shape:
        return ()

    multiplier = 1
    strides = [multiplier]
    for l in reversed(shape[1:]):
        multiplier = l * multiplier
        strides.append(multiplier)

    return tuple(reversed(strides))


def compute_reduction_output_shape(
    shape: ShapeType, dimensions: Sequence
) -> Tuple[int, ...]:
    for idx in dimensions:
        validate_idx(shape, idx)

    new_shape = []
    for idx in range(len(shape)):
        if idx in dimensions:
            continue

        new_shape.append(shape[idx])

    return tuple(new_shape)


def reduction_dims(shape: ShapeType, dims: Optional[Sequence]) -> Tuple[int, ...]:
    if dims is None:
        return tuple(range(len(shape)))
    dims = tuple(canonicalize_idx(shape, idx) for idx in dims)
    assert len(dims) == len(set(dims)), "duplicate value in dims"
    return dims
