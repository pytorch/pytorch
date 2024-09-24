# mypy: allow-untyped-defs
from __future__ import annotations

import operator
import warnings
from contextlib import nullcontext
from enum import Enum
from functools import reduce
from typing import (
    Any,
    Callable,
    cast,
    List,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import deprecated, TypeAlias


if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.

    import sympy

import torch
from torch import sym_float, sym_int, sym_max


ShapeType: TypeAlias = Union[torch.Size, List[int], Tuple[int, ...]]
StrideType: TypeAlias = Union[List[int], Tuple[int, ...]]
DimsType: TypeAlias = Union[int, List[int], Tuple[int, ...]]
DimsSequenceType: TypeAlias = Union[List[int], Tuple[int, ...]]
# TODO: Type[torch.SymInt], Type[torch.SymFloat]
NumberTypeType: TypeAlias = Union[Type[bool], Type[int], Type[float], Type[complex]]
# TODO: This needs a lot more type annotations
# NumberType = Union[bool, int, float, complex, torch.SymInt, torch.SymFloat]
NumberType: TypeAlias = Union[bool, int, float, complex]
RealNumberType: TypeAlias = Union[bool, int, float]

Number = (bool, int, float, complex, torch.SymInt, torch.SymFloat, torch.SymBool)
# I don't call it Integral because numbers.Integral includes bool, but IntLike
# does not
Dim = int
IntLike = (int, torch.SymInt)
FloatLike = (float, torch.SymFloat)
BoolLike = (bool, torch.SymBool)
IntWithoutSymInt = int
FloatWithoutSymFloat = float
DeviceLikeType: TypeAlias = Union[str, torch.device, int]
Tensor = torch.Tensor


torch_function_passthrough = {
    torch.device,
    torch.sym_not,
    torch.sym_float,
    torch.sym_int,
    torch.sym_max,
    torch.sym_min,
    torch._sym_sqrt,  # type: ignore[attr-defined]
    torch.sym_ite,
    torch.Tensor.dim,
    torch.Tensor.ndim.__get__,  # type: ignore[attr-defined]
    torch.Tensor.numel,
    torch.Tensor.size,
    torch.Tensor.storage_offset,
    torch.Tensor.stride,
    torch.Tensor.dtype.__get__,  # type: ignore[attr-defined]
    torch.Tensor.is_sparse.__get__,  # type: ignore[attr-defined]
    torch.Tensor.shape.__get__,  # type: ignore[attr-defined]
    torch.Tensor.device.__get__,  # type: ignore[attr-defined]
    torch.Tensor.requires_grad.__get__,  # type: ignore[attr-defined]
    torch.Tensor.layout.__get__,  # type: ignore[attr-defined]
    torch.Tensor.is_contiguous,
    # For TorchRefsMode only
    torch.Tensor.__format__,
    torch.Tensor.__repr__,
    torch.Tensor.requires_grad.__get__,  # type: ignore[attr-defined]
    torch.Tensor.__getitem__,
}


TensorLikeType = torch.Tensor
TensorLike = torch.Tensor
TensorSequenceType: TypeAlias = Union[List[TensorLikeType], Tuple[TensorLikeType, ...]]
TensorOrNumberLikeType: TypeAlias = Union[TensorLikeType, NumberType]

CustomOutParamAnnotation = "__custom_out_param__"


def same_shape(a: ShapeType, b: ShapeType, *, allow_rhs_unbacked=False) -> bool:
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if allow_rhs_unbacked:
            # TODO: We should check that the symbols are consistent
            # with each other
            if isinstance(y, torch.SymInt):
                continue
        # NB: Naively, you would not expect to have to do an oblivious guard
        # here because there is seemingly no broadcasting here, but in fact we
        # use this in some situations to determine if we need to do an expand
        # on the tensor because they don't line up, so you can definitely end
        # up trying to prove u0 != 1 in this situation.  See
        # python test/test_proxy_tensor.py -k test_cumsum_unbacked
        if guard_size_oblivious(x != y):
            return False

    return True


def _maybe_get_pytype(t):
    if t is torch.SymFloat:
        return float
    elif t is torch.SymInt:
        return int
    elif t is torch.SymBool:
        return bool
    else:
        return t


# TODO: look at using torch.testing.assert_close instead with an option
#   to just compare metadata
def compare_tensor_meta(
    a: TensorLikeType,
    b: TensorLikeType,
    check_strides=False,
    *,
    allow_rhs_unbacked=False,
    check_conj=True,
):
    """
    Checks that two tensor likes have the same shape,
    dtype and device.

    In the future this will validate additional metadata, like
    strides.
    """
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)

    if not same_shape(a.shape, b.shape, allow_rhs_unbacked=allow_rhs_unbacked):
        msg = f"Shapes {a.shape} and {b.shape} are not equal!"
        raise AssertionError(msg)

    if a.dtype != b.dtype:
        msg = f"Dtypes {a.dtype} and {b.dtype} are not equal!"
        raise AssertionError(msg)

    if a.device != b.device:
        # Handles special cuda:0 vs cuda case
        # TODO: we should review why this happens and see about fixing it
        if (str(a.device) == "cuda:0" or str(a.device) == "cuda") and (
            str(b.device) == "cuda:0" or str(b.device) == "cuda"
        ):
            pass
        else:
            msg = f"Devices {a.device} and {b.device} are not equal!"
            raise AssertionError(msg)

    # Stride checking is currently disabled, see https://github.com/pytorch/pytorch/issues/78050
    if check_strides:
        same_strides, idx = check_significant_strides(a, b)
        if not same_strides:
            msg = f"Stride mismatch! Strides are {a.stride()} and {b.stride()} (mismatched at {idx})!"
            raise RuntimeError(msg)

        if a.storage_offset() != b.storage_offset():
            msg = f"Storage offset mismatch! Storage offsets are {a.storage_offset()} and {b.storage_offset()}!"
            raise RuntimeError(msg)

    if check_conj:
        if a.is_conj() != b.is_conj():
            raise RuntimeError(
                f"Conj mismatch! is_conj is set to {a.is_conj()} and {b.is_conj()}"
            )

    if a.is_neg() != b.is_neg():
        raise RuntimeError(
            f"Neg mismatch! is_neg is set to {a.is_neg()} and {b.is_neg()}"
        )


def _check_strides_helper(
    a: TensorLikeType, b: TensorLikeType, *, only_cuda=True, significant_only=True
) -> Tuple[bool, Optional[int]]:
    # NOTE: only on CUDA because CPU elementwise strides are incorrect in PyTorch
    # See https://github.com/pytorch/pytorch/issues/77553
    # Only compares strides that are "meaningful" -- strides for dimensions with length > 1
    # and for tensors with more than one element
    if (
        not only_cuda or a.device.type == "cuda" or b.device.type == "cuda"
    ) and a.numel() > 0:
        for idx in range(a.ndim):
            check = not significant_only or a.shape[idx] > 1
            if a.stride()[idx] != b.stride()[idx] and check:
                return False, idx

    return True, None


def check_significant_strides(
    a: TensorLikeType, b: TensorLikeType, *, only_cuda=True
) -> Tuple[bool, Optional[int]]:
    return _check_strides_helper(a, b, only_cuda=only_cuda, significant_only=True)


def check_all_strides(
    a: TensorLikeType, b: TensorLikeType, *, only_cuda=True
) -> Tuple[bool, Optional[int]]:
    return _check_strides_helper(a, b, only_cuda=only_cuda, significant_only=False)


# This function is equivalent to compute_contiguous() from TensorImpl.cpp
def is_contiguous(a: TensorLikeType) -> bool:
    """
    Tests whether a tensor is contiguous or not.

    Tensors are contiguous when they have no elements,
    one element, or when they have "nested" strides.
    """
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    if guard_size_oblivious(a.numel() < 2):
        return True

    expected_stride = 1
    for x, y in reversed(tuple(zip(a.shape, a.stride()))):
        # Skips checking strides when a dimension has length 1
        if guard_size_oblivious(x == 1):
            continue

        if guard_size_oblivious(y != expected_stride):
            return False
        expected_stride = expected_stride * x

    return True


# This function is equivalent to compute_channels_last_contiguous_2d() in TensorImpl.cpp
def is_channels_last_contiguous_2d(a: Tensor) -> bool:
    # NHWC or not channels last 2D contiguous
    if a.ndim != 4:
        return False

    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    expected_stride = 1
    for idx in (1, 3, 2, 0):
        length = a.shape[idx]
        if guard_size_oblivious(length == 1):
            continue

        stride = a.stride()[idx]
        if guard_size_oblivious(stride != expected_stride):
            return False

        expected_stride *= length

    return True


def is_channels_last_contiguous_3d(a: Tensor) -> bool:
    # NDHWC or not channels last 3D contiguous
    if a.ndim != 5:
        return False

    expected_stride = 1
    for idx in (1, 4, 3, 2, 0):
        length = a.shape[idx]
        if length == 1:
            continue

        stride = a.stride()[idx]
        if stride != expected_stride:
            return False

        expected_stride *= length

    return True


_memory_formats = {
    torch.contiguous_format,
    torch.preserve_format,
    torch.channels_last,
    torch.channels_last_3d,
}


def validate_memory_format(memory_format: torch.memory_format):
    torch._check(
        memory_format in _memory_formats,
        lambda: f"Received unknown memory format {memory_format}!",
    )


def is_contiguous_for_memory_format(  # type: ignore[return]
    a: Tensor, *, memory_format: torch.memory_format
) -> bool:
    validate_memory_format(memory_format)

    if memory_format == torch.contiguous_format:
        return is_contiguous(a)
    if memory_format == torch.channels_last:
        return is_channels_last_contiguous_2d(a)
    if memory_format == torch.channels_last_3d:
        return is_channels_last_contiguous_3d(a)

    torch._check(
        False,
        lambda: f"is_contiguous received unsupported memory format {memory_format}",
    )


# NOTE: that tensors with no elements and channels last is ???
def is_channels_last_contiguous(a: Tensor) -> bool:
    """
    True when a tensor is channels-last contiguous.

    This requires that:

      - the tensor is conceptually either 4 (NHWC) or 5 (NDHWC) dimensions
      - if we name the tensor's dimensions NCHW or NCDHW, then the strides are such that the
        stride of the 'C' dimension (Cs) is 1 and the strides corresponding to
        each dimension (Xs) can be ordered Cs <= Ws <= Hs <= (Ds) <= Ns and are
        "nested" -- so Ws = Cs * Cl, where Cl is the length of the 'C' dimension,
        for example.
    """
    return is_channels_last_contiguous_2d(a) or is_channels_last_contiguous_3d(a)


def is_non_overlapping_and_dense(a: Tensor) -> bool:
    """
    True when a tensor is non-overlapping and dense.

    A tensor is non-overlapping and dense when there exists a permutation of
    its dimensions that is contiguous.
    """

    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    if a.is_sparse:
        return False

    # Short-circuits if the tensor is already contiguous or channels-last contiguous
    if is_contiguous(a) or is_channels_last_contiguous(a):
        return True

    # The following is equivalent to compute_non_overlapping_and_dense in TensorImpl.cpp

    # Short-circuits for tensors of rank one, which are
    # non-overlapping and "dense" if their stride is one
    if a.ndim == 1:
        return a.stride()[0] == 1

    # Checks that there exists a permutation of the strides s.t. the tensor would be contiguous
    # Sorts (length, stride) pairs by stride
    #
    # This sort is done in a size-oblivious way, which helps if we do a
    # comparison like 2048*u0 > u0; we just want this to return True
    # (and not worry about what if u0 is zero).
    class K(NamedTuple):
        size: int
        stride: int

        def __lt__(self, other):
            return guard_size_oblivious(self.stride < other.stride)

        def __gt__(self, other):
            return guard_size_oblivious(self.stride > other.stride)

        def __le__(self, other):
            return guard_size_oblivious(self.stride <= other.stride)

        def __ge__(self, other):
            return guard_size_oblivious(self.stride >= other.stride)

        def __eq__(self, other):
            return guard_size_oblivious(self.stride == other.stride)

    lengths_and_strides = sorted(map(K, a.shape, a.stride()))

    expected_stride = 1
    for length, stride in lengths_and_strides:
        if guard_size_oblivious(length == 1):
            continue

        if stride != expected_stride:
            return False

        expected_stride *= length

    return True


# NOTE: Based on the implementation in TensorIterator.cpp, but note that
# the note [Computing output strides] is incorrect, because it
# says that strides will be preserved even if they are not
# "non overlapping and dense", but this is incorrect. The
# output of elementwise operations are always given
# non overlapping and dense strides.
# This is also INCORRECT because it does not model TensorIterator's
# short-circuit, which can cause different strides.
def compute_elementwise_output_logical_to_physical_perm(
    *tensors, _skip_checks=False
) -> List[int]:
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    if not _skip_checks and len(tensors) == 0:
        msg = "Can't compute elementwise output strides for zero tensors!"
        raise ValueError(msg)

    if not _skip_checks:
        check_same_shape(*tensors, allow_cpu_scalar_tensors=True)

    # Filters the tensors to actual tensors
    if not _skip_checks:
        tensors = tuple(
            a
            for a in tensors
            if isinstance(a, TensorLike) and not is_cpu_scalar_tensor(a)
        )

    # Short-circuits for CPU scalar case
    if len(tensors) == 0:
        return []

    # Short-circuits for shapes with zero or one dimensions
    # TODO: are these necessary?
    ndim = tensors[0].ndim
    if ndim == 0:
        return []
    if ndim == 1:
        return [0]

    # Short-circuits if contiguous or channels last, following the fake fast path.
    # This reduces the number of guards we end up making
    is_contiguous = True
    is_channels_last = True
    for t in tensors:
        is_contiguous = is_contiguous and t.is_contiguous(
            memory_format=torch.contiguous_format
        )
        is_channels_last = is_channels_last and t.is_contiguous(
            memory_format=torch.channels_last
        )

    if is_contiguous and not is_channels_last:
        return list(range(ndim))

    if is_channels_last and not is_contiguous:
        return [0, *list(range(2, ndim)), 1]

    shape = tensors[0].shape

    def should_swap(idx_a, idx_b):
        for tensor in tensors:
            stride_a = tensor.stride()[idx_a]
            stride_b = tensor.stride()[idx_b]

            if guard_size_oblivious(stride_a == 0) or guard_size_oblivious(
                stride_b == 0
            ):
                continue

            if guard_size_oblivious(stride_a < stride_b):
                return -1

            if guard_size_oblivious(stride_a > stride_b):
                return 1

            # stride_a == stride_b
            if guard_size_oblivious(shape[idx_a] > shape[idx_b]):
                return 1

        # Note: this case is hit if all strides are zero,
        # or all strides are equal and all dimensions have the same length
        return 0

    # The "sort" order for the permutation is back-to-front, but
    # the natural order for permutations is front-to-back.  Do the
    # sorting back-to-front and then reverse it on output.
    #
    # also, note this returns the logical to physical shape permutation
    perm = list(reversed(range(ndim)))

    # insertion sort with support for ambiguous comparisons
    for i in range(1, ndim):
        dim1 = i
        for dim0 in reversed(range(i)):
            comparison = should_swap(perm[dim0], perm[dim1])
            if comparison > 0:
                perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
                dim1 = dim0
            elif comparison < 0:
                break

    return list(reversed(perm))


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

    ndim = tensors[0].ndim
    shape = tensors[0].shape

    if ndim == 0:
        return ()
    if ndim == 1:
        return (1,)

    logical_to_physical_perm = compute_elementwise_output_logical_to_physical_perm(
        *tensors, _skip_checks=True
    )
    permuted_shape = apply_perm(shape, logical_to_physical_perm)  # to physical

    new_strides = make_contiguous_strides_for(permuted_shape)
    permuted_strides = apply_perm(
        new_strides, invert_perm(logical_to_physical_perm)
    )  # to logical

    return tuple(permuted_strides)


# Identity permutation is [0, 1, 2]
def apply_perm(inp, perm):
    ndim = len(inp)
    permuted_inp = [-1] * ndim
    for idx, x in enumerate(perm):
        permuted_inp[idx] = inp[x]
    return permuted_inp


def invert_perm(perm):
    ndim = len(perm)
    new_perm = [-1] * ndim
    for idx, x in enumerate(perm):
        new_perm[x] = idx
    return new_perm


#
# Common helper functions
#


def validate_dim_length(length: int):
    """
    Validates that an object represents a valid
    dimension length.
    """

    if isinstance(length, (int, torch.SymInt)):
        torch._check_is_size(length)
    else:
        # sometimes called with sympy expression by inductor
        assert length >= 0


def validate_shape(shape: ShapeType):
    """
    Validates that a sequence represents a valid shape.
    """

    assert isinstance(shape, Sequence), type(shape)
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

    assert isinstance(idx, Dim)
    assert isinstance(rank, Dim)

    assert idx >= 0 and idx < rank or idx == 0


def validate_dimension_indices(rank: int, indices: DimsSequenceType):
    for idx in indices:
        validate_idx(rank, idx)


def validate_exclusive_idx(rank: int, ex_idx: int):
    """
    Validates that ex_idx is a valid exclusive index
    for the given shape.
    """

    assert isinstance(ex_idx, Dim)
    assert isinstance(rank, Dim)
    assert ex_idx > 0 and ex_idx <= rank


# "Wraps" a dim (up to one time) for the given rank, allowing dims to be
# specified using negative indices. If `wrap_scalar` is true then scalar
# tensors of rank 0 will allow dimensions in the range [-1, 0]. Otherwise,
# idx should be in the range [-rank, rank-1].
def canonicalize_dim(rank: int, idx: int, wrap_scalar: bool = True) -> int:
    if rank < 0:
        msg = f"Rank cannot be negative but got {rank}"
        raise IndexError(msg)

    if rank == 0:
        if not wrap_scalar:
            msg = f"Dimension specified as {idx} but tensor has no dimensions"
            raise IndexError(msg)
        rank = 1

    if idx >= 0 and idx < rank:
        return idx

    if idx < 0:
        _idx = idx + rank
    else:
        _idx = idx

    if _idx < 0 or _idx >= rank:
        # Same error message as in aten/src/ATen/WrapDimUtils.h:49
        msg = f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {idx})"
        raise IndexError(msg)

    return _idx


# Takes a dimension or sequence of dimensions and "wraps" them,
# mapping negative offsets to positive ones
@overload
def canonicalize_dims(
    rank: int, indices: Sequence[int], wrap_scalar: bool = True
) -> Tuple[int, ...]:
    pass


@overload
def canonicalize_dims(rank: int, indices: int, wrap_scalar: bool = True) -> int:
    pass


def canonicalize_dims(rank, indices, wrap_scalar=True):
    if isinstance(indices, Dim):
        return canonicalize_dim(rank, indices, wrap_scalar)

    return tuple(canonicalize_dim(rank, x, wrap_scalar) for x in indices)


def is_valid_permutation(rank: int, perm: DimsSequenceType) -> bool:
    """
    Validates that perm is a permutation of length rank.
    """

    return isinstance(perm, Sequence) and sorted(perm) == list(range(rank))


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
                msg = f"Shape {arg.shape} is not the expected shape {shape}!"
                raise RuntimeError(msg)
        else:
            msg = (
                "Unexpected type when checking for same shape, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)


# Acquires a common shape, if it exists, from one or more tensor arguments,
# filtering number arguments
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


# Extracts dimensions that might be passed either as a list/tuple or as varargs.
# A typical case is Tensor.permute .
def extract_dims_from_varargs(
    dims: Union[DimsSequenceType, Tuple[DimsSequenceType, ...]]
) -> DimsSequenceType:
    if dims and isinstance(dims[0], Sequence):
        assert len(dims) == 1
        dims = cast(Tuple[DimsSequenceType], dims)
        return dims[0]
    else:
        return cast(DimsSequenceType, dims)


def extract_shape_from_varargs(
    shape: Union[ShapeType, Tuple[ShapeType]],
    validate=True,
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
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = shape[0]

    if validate:
        validate_shape(shape)  # type: ignore[arg-type]
    return shape  # type: ignore[return-value]


def infer_size_shapes(a: ShapeType, b: ShapeType) -> Tuple[int, ...]:
    ndim = max(len(a), len(b))
    expandedSizes = [0] * ndim

    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = len(a) - 1 - offset
        dimB = len(b) - 1 - offset
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1

        torch._check(
            (sizeA == sizeB) or (sizeA == 1) or (sizeB == 1),
            lambda: (
                f"The size of tensor a ({sizeA}) must match the size of "
                f"tensor b ({sizeB}) at non-jagged dimension {i}"
            ),
        )

        # 1s map to the other size (even 0)
        expandedSizes[i] = sizeB if sizeA == 1 else sizeA

    return tuple(expandedSizes)


def infer_size(shape: ShapeType, numel: int) -> Tuple[int, ...]:
    """
    Infers the size of a dim with size -1, if it exists.
    Also checks that new shape is compatible with the number of elements.
    """
    dim = None
    newsize = 1
    for i, d in enumerate(shape):
        if d == -1:
            torch._check(dim is None, lambda: "only one dimension can be inferred")
            dim = i
        elif d >= 0:
            newsize *= d
        else:
            torch._check(False, lambda: f"invalid shape dimension {d}")
    if dim is None:
        torch._check(
            numel == newsize,
            lambda: f"shape '{list(shape)}' is invalid for input of size {numel}",
        )
    else:
        from torch.fx.experimental.symbolic_shapes import definitely_true

        torch._check(
            newsize != 0,
            lambda: (
                f"cannot reshape tensor of 0 elements into shape {list(shape)} because the "
                f"unspecified dimension size -1 can be any value and is ambiguous"
                if definitely_true(numel == 0)
                else f"shape '{list(shape)}' is invalid for input of size {numel}"
            ),
        )
        torch._check(
            numel % newsize == 0,
            lambda: f"shape '{list(shape)}' is invalid for input of size {numel}",
        )
        # Convert to list to produce a compatible error message with core
        # PyTorch, which prints sequences in square brackets.
        shape = list(shape)
        shape[dim] = numel // newsize
        # NB: This is pretty important when you have unbacked SymInts.
        # Suppose you have (i0, 12) resizing into (2, -1, 12).  The old
        # range for i0 is typically [2, inf], which means if you divide
        # by two the new range should be [1, inf].  But this is bad news
        # if you have an unbacked SymInt: we need to reapply the unsound
        # assumption that the size is >= 2.
        torch._check_is_size(shape[dim])
    return tuple(shape)


_integer_dtypes = (
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_low_precision_dtypes = (torch.float16, torch.bfloat16, torch.complex32)
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
    return dtype.is_floating_point


def is_complex_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype in _complex_dtypes


def is_grad_dtype(dtype: torch.dtype) -> bool:
    """
    Checks if the dtype can require a gradient.
    """
    return dtype.is_floating_point or is_complex_dtype(dtype)


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
    if dtype.is_floating_point:
        return float
    if dtype in _complex_dtypes:
        return complex

    raise ValueError("Invalid dtype!")


def dtype_to_type_ctor(dtype: torch.dtype) -> Callable[[NumberType], NumberType]:
    """
    Computes the corresponding Python type constructor for the
    given dtype.
    """
    assert isinstance(dtype, torch.dtype)

    if dtype is torch.bool:
        return lambda x: bool(x)
    if dtype in _integer_dtypes:
        return sym_int
    if dtype.is_floating_point:
        return sym_float
    if dtype in _complex_dtypes:
        # TODO: type error here is real, replace with sym_complex
        return lambda x: complex(x)  # type: ignore[arg-type]

    raise ValueError("Invalid dtype!")


def type_to_dtype(typ: type) -> torch.dtype:
    """
    Computes the corresponding dtype for a Number type.
    """

    assert isinstance(typ, type)

    if typ in (bool, torch.SymBool):
        return torch.bool
    if typ in (int, torch.SymInt):
        return torch.long
    if typ in (float, torch.SymFloat):
        return torch.get_default_dtype()
    # TODO: sym_complex_float?
    if typ is complex:
        return corresponding_complex_dtype(torch.get_default_dtype())

    raise ValueError(f"Invalid type {typ}!")


def get_dtype(x: Union[torch.Tensor, NumberType]):
    if isinstance(x, torch.Tensor):
        return x.dtype
    else:
        return type_to_dtype(type(x))


_ordered_types = (bool, int, float, complex)


def check_fp_or_complex(
    dtype: torch.dtype, fn_name: str, allow_low_precision_dtypes: bool = True
):
    """
    Checks whether the input is floating point or complex.
    If allow_low_precision_dtypes is True, it allows having float16, bfloat16, and complex32
    """
    torch._check(
        is_float_dtype(dtype) or is_complex_dtype(dtype),
        lambda: f"{fn_name}: Expected a floating point or complex tensor as input. Got {dtype}",
    )
    torch._check(
        allow_low_precision_dtypes or not is_low_precision_dtype(dtype),
        lambda: f"{fn_name}: Half precision dtypes not supported. Got {dtype}",
    )


def check_is_matrix(A: TensorLikeType, f_name: str, arg_name: str = "A"):
    torch._check(
        len(A.shape) >= 2,
        lambda: f"{f_name}: The input tensor {arg_name} must have at least 2 dimensions.",
    )


def get_higher_type(a: type, b: type) -> type:
    """
    Returns the higher of the two given Number types.

    The types are ordered bool -> int -> float -> complex.
    """
    a, b = _maybe_get_pytype(a), _maybe_get_pytype(b)
    # Type checking
    if a not in _ordered_types or b not in _ordered_types:
        raise RuntimeError(f"Expected builtin numeric types, found {a}, {b}")

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


def check_pin_memory(pin_memory: bool):
    torch._check_not_implemented(
        not pin_memory, lambda: "PrimTorch does not support pinned memory"
    )


def check_layout(layout: torch.layout):
    torch._check_not_implemented(
        layout == torch.strided, lambda: f"PrimTorch doesn't support layout={layout}"
    )


# TODO: maybe unify with can_cast_to?
def is_weakly_lesser_type(a: type, b: type) -> bool:
    """
    Compares two types, a and b, returning True if a is weakly "less" than b.

    The comparison is determined by the following type ordering: bool, int, float, complex.
    """

    a, b = _maybe_get_pytype(a), _maybe_get_pytype(b)

    if a not in _ordered_types or b not in _ordered_types:
        raise RuntimeError(f"Expected builtin numeric types, found {a}, {b}")

    for typ in _ordered_types:
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

    raise ValueError(f"Received unknown dtypes {cast_to}, {cast_from}!")


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


_cpu_acc_type_map = {
    torch.bfloat16: torch.float64,
    torch.float16: torch.float64,
    torch.float32: torch.float64,
    torch.complex32: torch.complex128,
    torch.complex64: torch.complex128,
}


def get_acc_type(dtype: torch.dtype, device: torch.device) -> torch.dtype:
    # Equivalent to at::toAccumulateType, prefer computation_dtype where possible
    if device.type == "cpu":
        return _cpu_acc_type_map.get(dtype, dtype)
    else:
        return get_computation_dtype(dtype)


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
    NONE = (3,)


# TODO: when NumberType contains the sym types, can simplify this
def number_type(
    x: Union[NumberType, torch.SymInt, torch.SymFloat, torch.SymBool]
) -> Type:
    if isinstance(x, torch.SymInt):
        return int
    elif isinstance(x, torch.SymFloat):
        return float
    elif isinstance(x, torch.SymBool):
        return bool
    else:
        return type(x)


def expr_type(x: sympy.Basic) -> Type:
    import sympy

    if x.kind is sympy.core.kind.BooleanKind:
        return bool
    elif x.is_integer:  # type: ignore[attr-defined]
        return int
    else:
        # NB: Not strictly correct, but we don't support SymPy complex or bool.
        return float


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

    The INT_TO_FLOAT type promotion kind maps boolean and integer result dtypes to the default floating point dtype,
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

    # Import sympy locally, as importing it eagerly at a module level is too slow
    # See https://dev-discuss.pytorch.org/t/delving-into-what-happens-when-you-import-torch/1589
    import sympy

    for x in args:
        if not isinstance(x, (Number, TensorLike, sympy.Basic)):
            msg = f"Unexpected type {str(type(x))} when computing elementwise type promotion!"
            raise ValueError(msg)

        if isinstance(x, Number):
            highest_type = get_higher_type(highest_type, number_type(x))
        elif isinstance(x, sympy.Basic):
            highest_type = get_higher_type(highest_type, expr_type(x))
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
        raise ValueError(f"Unknown type promotion kind {str(type_promotion_kind)}")


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


# This function's logic is borrowed from the following functions defined in C++:
# batched_matrix_contiguous_strides and contiguous_strides
def make_contiguous_strides_for(
    shape: ShapeType, row_major: bool = True
) -> Tuple[int, ...]:
    """
    Returns the strides of a contiguous tensor if row_major
    If row_major=True, it returns the strides of a contiguous batch of Fortran-contiguous matrices
    This is often used when calling external libraries like BLAS/LAPACK/cuSolver...
    """
    # contiguous_strides from c10/util/strides.h
    validate_shape(shape)
    if not shape:
        return ()

    from torch.fx.experimental.symbolic_shapes import is_nested_int

    multiplier = 1
    strides = []
    for l in reversed(shape):
        strides.append(multiplier)
        multiplier *= l if is_nested_int(l) else sym_max(l, 1)

    result = tuple(reversed(strides))

    # batched_matrix_contiguous_strides from aten/src/ATen/native/LinearAlgebraUtils.h
    if row_major:
        return result
    else:
        if len(shape) < 2:
            return result
        return result[:-2] + (1, max(shape[-2], 1))


def make_channels_last_1d_strides_for(shape: ShapeType) -> Tuple[int, ...]:
    torch._check(
        len(shape) == 3,
        lambda: "Only tensors of rank 3 can use the channels_last_1d memory format",
    )

    multiplier = 1
    strides = [0] * 3
    for idx in (1, -1, 0):
        # NOTE: intentionally divergence from make_contiguous_strides_for
        # This is consistent with eager
        strides[idx] = multiplier
        multiplier *= shape[idx]

    return tuple(strides)


def make_channels_last_2d_strides_for(shape: ShapeType) -> Tuple[int, ...]:
    # TODO: maybe inform the user of channels_last_3d if rank of the tensor is 5?
    torch._check(
        len(shape) == 4,
        lambda: "Only tensors of rank 4 can use the channels_last memory format",
    )

    multiplier = 1
    strides = [0] * 4
    for idx in (1, -1, -2, 0):
        # NOTE: intentionally divergence from make_contiguous_strides_for
        # This is consistent with eager
        strides[idx] = multiplier
        multiplier *= shape[idx]

    return tuple(strides)


def make_channels_last_3d_strides_for(shape: ShapeType) -> Tuple[int, ...]:
    torch._check(
        len(shape) == 5,
        lambda: "Only tensors of rank 5 can use the channels_last_3d memory format",
    )

    multiplier = 1
    strides = [0] * 5
    for idx in (1, -1, -2, -3, 0):
        # NOTE: intentionally divergence from make_contiguous_strides_for
        # This is consistent with eager
        strides[idx] = multiplier
        multiplier *= shape[idx]

    return tuple(strides)


def make_channels_last_strides_for(shape: ShapeType) -> Tuple[int, ...]:
    ndim = len(shape) if isinstance(shape, Sequence) else 1
    if ndim == 3:
        return make_channels_last_1d_strides_for(shape)
    elif ndim == 4:
        return make_channels_last_2d_strides_for(shape)
    elif ndim == 5:
        return make_channels_last_3d_strides_for(shape)
    else:
        raise RuntimeError(
            f"no channels last format strides exist in {ndim} dimensions"
        )


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


def set_correction(
    unbiased: Optional[bool] = None,
    correction: Optional[NumberType] = None,
) -> float:
    if correction is not None and unbiased is not None:
        raise RuntimeError("cannot specify both correction and unbiased arguments")
    elif correction is None and unbiased is None:
        correction = 1.0
    elif correction is None and unbiased is not None:
        correction = 0.0 if unbiased is False else 1.0
    # NB: we don't actually support symint here, but it's harmless to accept
    if not isinstance(correction, (IntLike, FloatLike)):
        raise ValueError("correction argument should be integer or float")
    if correction < 0:
        raise ValueError("correction argument should be non-negative")
    return sym_float(correction)


def compute_required_storage_length(
    shape: ShapeType, strides: StrideType, storage_offset: int
) -> int:
    """Computes the minimum storage size to hold the given tensor geometry.

    Example
    =======

    This is the size of a newly allocated tensor's storage, in units of elements

    >>> t = torch.empty((10, 20))
    >>> compute_required_storage_length(t.shape, t.stride(), t.storage_offset())
    200

    >>> # xdoctest: +SKIP(failing)
    >>> t2 = torch.empty_strided((1, 2, 3), (5, 7, 11))
    >>> size = compute_required_storage_length(t2.shape, t2.stride(), t2.storage_offset())
    >>> size == t.storage().size()
    True

    A valid tensor may have a larger storage size, but never smaller

    >>> slice = torch.empty(100)[20:40]
    >>> slice.storage().size()
    100

    >>> compute_required_storage_length(slice.shape, slice.stride(), slice.storage_offset())
    40

    """
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # Short-circuits if the shape has no elements
    if guard_size_oblivious(reduce(operator.mul, shape, 1) == 0):
        return 0

    max_offset = sum((x - 1) * y for x, y in zip(shape, strides))
    # +1 to account for the first element which offsets are taken from
    return 1 + storage_offset + max_offset


def check_in_bounds_for_storage(
    a: torch.TypedStorage, shape: ShapeType, strides: StrideType, storage_offset: int
):
    """
    Determines if the given shape, strides, and offset are valid for the given storage.
    """

    required_length = compute_required_storage_length(shape, strides, storage_offset)
    if a.size() < required_length:
        msg = (
            f"Can't view a storage of size {a.size()} with an offset of {storage_offset}, "
            f"shape of {str(shape)}, and strides of {str(strides)}, "
            f"which requires a storage of size {required_length}"
        )
        raise ValueError(msg)


# NOTE: This function should ideally be removed, but some Meta internal models
# packaged with `torch.package` are using it, so it will have to be removed
# at some point in the future when those models no longer use this function.
@deprecated(
    "`torch._prims_common.check` is deprecated and will be removed in the future. "
    "Please use `torch._check*` functions instead.",
    category=FutureWarning,
)
def check(
    b: bool, s: Callable[[], str], exc_type: Type[Exception] = RuntimeError
) -> None:
    """
    Helper function for raising an error_type (default: RuntimeError) if a boolean condition fails.
    Error message is a callable producing a string (to avoid wasting time
    string formatting in non-error case, and also to make it easier for torchdynamo
    to trace.)

    .. note:: This function is planned for removal in the future. Please use
        `torch._check*` functions instead.
    """
    torch._check_with(exc_type, b, s)


# This combines is_channels_last_strides_2d and is_channels_last_strides_3d in
# c10/core/MemoryFormat.h into one function
def are_strides_like_channels_last(
    shape: Sequence[int], strides: Sequence[int]
) -> bool:
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    ndim = len(shape)

    if ndim == 4:
        # Check for channels_last_2d
        dim_order = [1, 3, 2, 0]
    elif ndim == 5:
        # Check for channels_last_3d
        dim_order = [1, 4, 3, 2, 0]
    else:
        return False

    if guard_size_oblivious(strides[1] == 0):
        return False

    min = 0
    for d in dim_order:
        if guard_size_oblivious(shape[d] == 0):
            return False
        if guard_size_oblivious(strides[d] < min):
            return False
        if d == 0 and min == strides[1]:
            return False
        min = strides[d]
        if guard_size_oblivious(strides[d] > 1):
            min *= shape[d]
    return True


def suggest_memory_format(x: TensorLikeType) -> torch.memory_format:
    if x.layout != torch.strided:
        return torch.contiguous_format

    if are_strides_like_channels_last(x.shape, x.stride()):
        return torch.channels_last if x.ndim == 4 else torch.channels_last_3d

    return torch.contiguous_format


def prod(xs: Sequence[NumberType]) -> NumberType:
    """Product of elements in input sequence. Returns 1 for empty sequence"""
    return reduce(operator.mul, xs, 1)


def is_expandable_to(shape: ShapeType, desired: ShapeType) -> bool:
    """Checks if a shape can be expanded to another shape.
    This is equivalent to checking if the two shapes are broadcastable.
    """
    # This is a Python implementation of
    # aten/src/ATen/ExpandUtils.h:is_expandable_to
    if len(shape) > len(desired):
        return False
    for i in range(len(shape)):
        if shape[-i - 1] != desired[-i - 1] and shape[-i - 1] != 1:
            return False
    return True


def mask_tensor(mask: TensorLikeType, t: TensorLikeType):
    """
    Similar to torch.where(mask, t, 0) but if t is boolean,
    result is also boolean and not promoted to int.
    """
    # torch.where(mask, t, False) is equivalent
    # but feels hacky and might break in the future
    if t.dtype is torch.bool:
        return mask.logical_and(t)
    else:
        return torch.where(mask, t, 0)


def get_aten_op(fn: Callable, name: str):
    """
    Given the __module__ of reference and its name, it returns
    (our best guess of) the ATen name of the associated operation

    Note: In ATen, the __name__ of a function within a module often
    starts by the module name. E.g. linalg_eigh, or special_zeta
    """
    module = fn.__module__
    prefix = "torch._refs"
    assert module.startswith(prefix)
    module = module[len(prefix) :]
    # We want to go from .special / .nn.functional
    # to special and special_ / nn_functional_
    if module:
        module = module[1:]
        module = module.replace(".", "_")
        module = module + "_"
    return getattr(torch._ops.ops.aten, f"{module}{name}")


def dtype_or_default(dtype: Optional[torch.dtype]) -> torch.dtype:
    return dtype if dtype is not None else torch.get_default_dtype()


def device_or_default(device: Optional[DeviceLikeType]) -> DeviceLikeType:
    return device if device is not None else torch.device("cpu")


def layout_or_default(layout: Optional[torch.layout]) -> torch.layout:
    return layout if layout is not None else torch.strided


def clone_preserve_strides(x):
    needed_size = compute_required_storage_length(
        x.size(), x.stride(), x.storage_offset()
    )
    # Our eager implementations for *_scatter ops are all primitives w.r.t autograd,
    # so these as_strided() calls are not seen by autograd.
    # We need to mimic this behavior in our ref/prim implementations.
    # TODO: a better way to handle this would be with a new op, "_unsafe_as_strided"
    # We should revisit this when we add a compositional as_strided op,
    # and also as part of https://github.com/pytorch/pytorch/issues/90507
    try:
        old = torch._C._dispatch_tls_is_dispatch_key_excluded(
            torch._C.DispatchKey.ADInplaceOrView
        )
        torch._C._dispatch_tls_set_dispatch_key_excluded(
            torch._C.DispatchKey.ADInplaceOrView, True
        )
        buffer = torch.as_strided(x, (needed_size,), (1,), 0).clone()
        return torch.as_strided(buffer, x.size(), x.stride(), x.storage_offset())
    finally:
        torch._C._dispatch_tls_set_dispatch_key_excluded(
            torch._C.DispatchKey.ADInplaceOrView, old
        )


def alert_not_deterministic(caller: str):
    if torch.are_deterministic_algorithms_enabled():
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(
                f"{caller} does not have a deterministic implementation, but you set "
                f"'torch.use_deterministic_algorithms(True, warn_only=True)'. "
                f"You can file an issue at https://github.com/pytorch/pytorch/issues "
                f"to help us prioritize adding deterministic support for this operation."
            )
        else:
            torch._check(
                False,
                lambda: (
                    f"{caller} does not have a deterministic implementation, but you set "
                    f"'torch.use_deterministic_algorithms(True)'. You can turn off "
                    f"determinism just for this operation, or you can use the "
                    f"'warn_only=True' option, if that's acceptable for your application. "
                    f"You can also file an issue at https://github.com/pytorch/pytorch/issues "
                    f"to help us prioritize adding deterministic support for this operation."
                ),
            )


class CUDARngStateHelper:
    @staticmethod
    def get_torch_state_as_tuple(fake_mode=nullcontext()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        with fake_mode:
            seed = torch.tensor(torch.cuda.initial_seed())
            offset = torch.tensor(torch.cuda._get_rng_state_offset())
            return seed, offset

    @staticmethod
    def set_torch_state_tensor(seed, offset):
        # Rng state is [64-bit seed, 64-bit offset]
        seed_portion = seed.reshape([1]).view(torch.uint8)
        offset_portion = offset.reshape([1]).view(torch.uint8)
        new_state = torch.cat([seed_portion, offset_portion])
        torch.cuda.set_rng_state(new_state)

    @staticmethod
    def set_new_offset(relative_offset):
        torch.cuda._set_rng_state_offset(relative_offset.item())
