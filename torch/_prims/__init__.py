import torch
from torch._C import _add_docstr  # type: ignore[attr-defined]

from typing import Sequence, Optional
from numbers import Number
from functools import reduce


__all__ = [
  # Elementwise unary operators
  'abs',
  'acos',
  'acosh',
  'asin',
  'atan',
  'cos',
  'cosh',
  # Elementwise binary operators
  'add',
  'atan2',
  'div',
  'mul',
  'sub',
  # View operators
  'broadcast',  # sugar
  'broadcast_in_dim',
  'merge_dims',
  'split_dim',
  'squeeze',
  # Shape operators
  'collapse',  # sugar
  'concatenate',
  'reshape',
  # Type conversion operators
  'convert_element_type',
  'device_put',
  # Inplace operators
  'copy_to',
]

Tensor = torch.Tensor

#
# Common helper functions
#

def _validate_dim_length(length: int):
  '''
  Validates that an object represents a valid
  dimension length.
  '''

  assert isinstance(length, int)
  assert length >= 0

def _validate_shape(shape: Sequence):
  '''
  Validates that a sequence represents a valid shape.
  '''

  assert isinstance(shape, Sequence)
  for l in shape:
    _validate_dim_length(l)

def _validate_idx(shape: Sequence, idx: int):
  '''
  Validates that idx is a valid idx for the given shape.
  '''

  assert isinstance(idx, int)
  assert idx >= 0 and idx < len(shape)

def _validate_exclusive_idx(shape: Sequence, ex_idx: int):
  '''
  Validates that ex_idx is a valid exclusive index
  for the given shape.
  '''

  assert isinstance(ex_idx, int)
  assert ex_idx > 0 and ex_idx <= len(shape)

def _validate_permutation(rank: int, perm: Sequence):
  '''
  Validates that perm is a permutation of length rank.
  '''

  assert isinstance(perm, Sequence)
  assert tuple(sorted(perm)) == tuple(range(0, rank))

def _dtype_to_kind(dtype: torch.dtype):
  '''
  Computes the corresponding Python type (AKA "type kind") for the
  given dtype.
  '''
  assert isinstance(dtype, torch.dtype)

  bools = (torch.bool,)
  ints = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
  floats = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
  complexes = (torch.complex64, torch.complex128)

  if dtype in bools:
    return bool
  if dtype in ints:
    return int
  if dtype in floats:
    return float
  return complex

# TODO: refactor into util
def _higher_type_kind(a: type, b: type):
  '''
  Returns the higher of the two given type kinds.

  Type kinds are ordered bool -> int -> float -> complex.
  '''
  # Type checking
  assert isinstance(a, type)
  assert isinstance(b, type)

  if a is b:
      return a

  ordered_type_kinds = (
      bool, int, float, complex
  )

  for kind in ordered_type_kinds:
      if a is kind:
          return b
      if b is kind:
          return a

  raise ValueError("Unknown kind!")

# TODO: support additional datastructures
def _check_same_device(*args, allow_scalars):
  if len(args) <= 1:
    return

  # Note: cannot initialize device to the first arg's device (it may not have one)
  device = None
  for arg in args:
    if isinstance(arg, Number):
      assert allow_scalars, "Found a scalar when checking for same device but scalars not allowed!"
    elif isinstance(arg, Tensor):
      if device is None:
        device = arg.device

      assert device == arg.device, "Tensor on device " + str(arg.device) + " is not on the expected device " + str(device) + "!"
    else:
      assert False, "Unexpected type when checking for same device, " + str(type(arg)) + "!"

# Asserts if any of the following are true:
#   - a non-scalar or non-Tensor is given
#   - the shape of any tensors is distinct
# TODO: change asserts to RuntimeError and ValueError exceptions
# TODO: update to handle containers
def _check_same_shape(*args):
  shape = None

  for arg in args:
    if isinstance(arg, Number):
      continue
    elif isinstance(arg, Tensor):
      if shape is None:
        shape = arg.shape

      assert tuple(shape) == tuple(arg.shape)
    else:
      assert False, "Unexpected type when checking for same shape, " + str(type(arg)) + "!"

# Returns the Python type (AKA "type kind") corresponding to the torch dtype.
def _dtype_to_kind(dtype):
  bools = (torch.bool,)
  ints = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
  floats = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
  complexes = (torch.complex64, torch.complex128)

  if dtype in bools:
    return bool
  if dtype in ints:
    return int
  if dtype in floats:
    return float
  return complex

# Asserts if any of the following are true:
#   - the dtypes of any tensor args are different
#   - the type kind of any args are different
#   - a non-scalar or non-Tensor is given
# TODO: change asserts to RuntimeError and ValueError exceptions
# TODO: update to handle containers
def _check_same_dtype(*args):
  full_dtype = None
  scalar_dtype = None

  for arg in args:
    if isinstance(arg, Number):
      if scalar_dtype is None:
        scalar_dtype = type(arg)

      assert scalar_dtype is type(arg), "Scalar of type " + str(type(arg)) + " is not the expected dtype of " + str(scalar_dtype) + "!"
    elif isinstance(arg, Tensor):
      if full_dtype is None:
        full_dtype = arg.dtype
      if scalar_dtype is None:
        scalar_dtype = _dtype_to_kind(arg.dtype)

      assert full_dtype is arg.dtype, "Tensor with dtype " + str(arg.dtype) + " is not the expected dtype of " + str(full_dtype) + "!"
      assert _dtype_to_kind(arg.dtype) is scalar_dtype, "Tensor with dtype " + str(arg.dtype) + " is not the expected dtype kind of " + str(scalar_dtype) + "!"
    else:
      assert False, "Unexpected type when checking for same dtype, " + str(type(arg)) + "!"

#
# Elementwise operations
#

# TODO: extend to handle scalar x scalar cases consistently

def _elementwise_checks(*args, allow_scalars=True):
  _check_same_device(*args, allow_scalars=allow_scalars)
  _check_same_dtype(*args)
  _check_same_shape(*args)

def _make_elementwise_unary_op(op, *, allow_scalars=False):
  def _op(a):
    _elementwise_checks(a, allow_scalars=allow_scalars)
    return op(a)
  return _op

def _make_elementwise_binary_op(op, *, allow_scalars=True):
  def _op(a, b):
    _elementwise_checks(a, b, allow_scalars=allow_scalars)
    return op(a, b)
  return _op


#
# Elementwise unary operations
#

abs = _make_elementwise_unary_op(torch.abs)

# TODO: make sugar
acos = _make_elementwise_unary_op(torch.acos)

acosh = _make_elementwise_unary_op(torch.acosh)

# TODO: make sugar
asin = _make_elementwise_unary_op(torch.asin)

# TODO: make sugar
atan = _make_elementwise_unary_op(torch.atan)

cos = _make_elementwise_unary_op(torch.cos)

cosh = _make_elementwise_unary_op(torch.cosh)


#
# Elementwise binary operations
#

add = _make_elementwise_binary_op(torch.add)

atan2 = _make_elementwise_binary_op(torch.atan2)

div = _make_elementwise_binary_op(torch.true_divide)

mul = _make_elementwise_binary_op(torch.mul)

sub = _make_elementwise_binary_op(torch.sub)


# View operations

def _broadcast_in_dim_checks(
  a: Tensor,
  shape: Sequence[int],
  broadcast_dimensions: Sequence[int]):
    # Type checks
    assert isinstance(a, Tensor)
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

def _broadcast_in_dim_aten(a, shape, broadcast_dimensions):
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    v = a
    for idx, x in enumerate(s):
        if x != - 1:
            v.unsqueeze_(idx)

    return v.expand(shape)


def broadcast_in_dim(a, shape, broadcast_dimensions):
    '''
    Creates a view of t with the specified shape.

    Allows adding dimensions of any length and broadcasting
    dimensions of length one in t to any length.

    The location of the broadcast dimensions must be specified
    using the broadcast_dimensions argument. Changing the
    relative order of dimensions is not supported.
    '''

    _broadcast_in_dim_checks(a, shape, broadcast_dimensions)
    return _broadcast_in_dim_aten(a, shape, broadcast_dimensions)

def broadcast(a, leading_lengths):
  '''
  A wrapper around broadcast_in_dim that just adds outermost dimensions
  to a.
  '''
  shape = leading_lengths + a.shape
  rank = len(leading_lengths) + len(a.shape)
  return broadcast_in_dim(a, shape, tuple(range(len(leading_lengths), rank)))

def _merge_dims_checks(a: Tensor, start: int, end: int):
  assert isinstance(a, Tensor)

  shape = a.shape
  strides = a.stride()

  _validate_idx(shape, start)
  _validate_exclusive_idx(shape, end)

  # Verifies end greater than start
  assert end > start

  for idx in range(start, end - 1):
    assert strides[idx] == strides[idx + 1] * shape[idx + 1]

def _merge_dims_aten(a: Tensor, start: int, end: int) -> Tensor:
  # Short-circuits on null op
  if start == end - 1:
    return a

  dim_length = 1
  for idx in range(start, end):
    dim_length = dim_length * a.shape[idx]

  new_shape = a.shape[0:start] + (dim_length,) + a.shape[end:]

  return a.view(new_shape)

def merge_dims(a: Tensor, start: int, end: int) -> Tensor:
  '''
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
  '''

  _merge_dims_checks(a, start, end)
  return _merge_dims_aten(a, start, end)

def _split_dim_checks(a: Tensor, dim: int, outer_length: int):
  assert isinstance(a, Tensor)
  assert isinstance(dim, int)
  assert isinstance(outer_length, int)

  # Verifies dim is a valid idx
  assert dim >=0 and dim < len(a.shape)

  # Verifies outer_length is a valid length
  assert outer_length >= 0

  # Verifies the dim can be split with the specified lhs_length
  inner_length = a.shape[dim] / outer_length
  assert int(inner_length) == inner_length

def _split_dim_aten(a: Tensor, dim: int, outer_length: int) -> Tensor:
  inner_length = int(a.shape[dim] / outer_length)
  new_shape = a.shape[0:dim] + (outer_length, inner_length) + a.shape[dim + 1:]

  return a.view(new_shape)

def split_dim(a: Tensor, dim: int, outer_length: int) -> Tensor:
  '''
  Creates a view of a with the given dimension (of length l) split
  into two dimensions, with the outer of the two having
  length outer_length and the inner of the two having computed
  length inner_length such outer_length * inner_length = l.
  '''

  _split_dim_checks(a, dim, outer_length)
  return _split_dim_aten(a, dim, outer_length)

# Note: allows dimensions to be specified redundantly
def _squeeze_checks(a: Tensor, dimensions: Sequence):
  assert isinstance(a, Tensor)

  for idx in dimensions:
    _validate_idx(a.shape, idx)
    assert a.shape[idx] == 1

def _squeeze_aten(a: Tensor, dimensions: Sequence) -> Tensor:
  for idx in dimensions:
    a = torch.squeeze(a, dim=idx)

  return a

def squeeze(a: Tensor, dimensions: Sequence) -> Tensor:
  '''
  Creates a view of a with the specified dimensions of
  length one removed.
  '''

  _squeeze_checks(a, dimensions)
  return _squeeze_aten(a, dimensions)

#
# Shape operations
#
def collapse(a: Tensor, start: int, end: int) -> Tensor:
  '''
  Wrapper around reshape that collapses a span of dimensions.

  See merge_dims for the corresponding view operation.
  '''

  dim_length = 1
  for idx in range(start, end):
    dim_length = dim_length * a.shape[idx]

  new_shape = a.shape[0:start] + (dim_length,) + a.shape[end:]
  return reshape(a, new_shape)

def _concatenate_checks(tensors: Sequence[Tensor], dim: int):
  assert len(tensors) > 0
  assert dim >= 0

  shape = tensors[0].shape
  assert dim < len(shape)

  _check_same_dtype(*tensors)

  # Verifies same shape (except in the concat dimension)
  for tensor in tensors:
    for idx, lengths in enumerate(zip(shape, tensor.shape)):
      common_length, length = lengths
      if idx == dim:
        continue
      assert length == common_length

def _concatentate_aten(tensors: Sequence[Tensor], dim: int) -> Tensor:
  return torch.cat(tensors, dim)

def concatenate(tensors: Sequence[Tensor], dim: int) -> Tensor:
  _concatenate_checks(tensors, dim)
  return _concatentate_aten(tensors, dim)

def _reshape_checks(a: Tensor, shape: Sequence):
  assert isinstance(a, Tensor)
  _validate_shape(shape)

  # Validates the tensor and the requested shape have the
  # same number of elements
  numel = reduce(lambda acc, x: acc * x, shape)
  assert a.numel() == numel

def _reshape_aten(a: Tensor, shape: Sequence) -> Tensor:

  return a.clone().reshape(shape).contiguous()

def reshape(a: Tensor, shape: Sequence) -> Tensor:
  '''
  Creates a contiguous tensor with the specified shape
  containing a copy of the data in a.
  '''

  _reshape_checks(a, shape)
  return _reshape_aten(a, shape)

#
# Type conversions
#

def _convert_element_type_checks(a: Tensor, dtype: torch.dtype):
  # Type checks
    assert isinstance(a, Tensor)
    assert isinstance(dtype, torch.dtype)

def _convert_element_type_aten(a: Tensor, dtype: torch.dtype) -> Tensor:
  return a.to(dtype)

def convert_element_type(a: Tensor, dtype: torch.dtype) -> Tensor:
  _convert_element_type_checks(a, dtype)
  return _convert_element_type_aten(a, dtype)

def _device_put_checks(a: Tensor, device):
  assert isinstance(a, Tensor)
  assert isinstance(device, (str, torch.device))

def _device_put_aten(a: Tensor, device) -> Tensor:
  return a.to(device)

def device_put(a: Tensor, device) -> Tensor:
  _device_put_checks(a, device)
  return _device_put_aten(a, device)

#
# Inplace operators
#

def _copy_to_checks(a: Tensor, b: Tensor):
  assert isinstance(a, Tensor)
  assert isinstance(b, Tensor)

  # Validates the cast is safe
  a_kind = _dtype_to_kind(a.dtype)
  b_kind = _dtype_to_kind(b.dtype)
  assert a_kind is _higher_type_kind(a_kind, b_kind)

  assert a.numel() == b.numel()

def _copy_to_aten(a: Tensor, b: Tensor) -> Tensor:
  return a.copy_(b)

# TODO: implementing 'casting' options like NumPy
# Currently implements 'safe' casting
def copy_to(a: Tensor, b: Tensor) -> Tensor:
  '''
  Copies the data in b to a inplace and returns the modified a.
  '''

  _copy_to_checks(a, b)
  return _copy_to_aten(a, b)
