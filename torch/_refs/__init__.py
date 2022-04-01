import torch
from torch._C import _add_docstr  # type: ignore[attr-defined]

import torch._prims as prims
from functools import reduce

from numbers import Number
from typing import Sequence, Optional

all = [
  'add',
]

Tensor = torch.Tensor

# Returns the Python type (AKA "type kind") corresponding to the torch dtype.
# TODO: refactor into utils
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
  if dtype in complexes:
    return complex

  raise ValueError("Invalid dtype!")

# Returns the higher of two torch datatypes a and b or, if the two
#   are not ordered relative to each other, the next
#   higher datatype
def _higher_dtype(a: torch.dtype, b: torch.dtype):
  '''
  Computes the "lowest" datatype that is weakly
  "higher" than both a and b.
  '''

  # Type checking
  assert isinstance(a, torch.dtype)
  assert isinstance(b, torch.dtype)

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
    (torch.complex128,)
  )

  for idx, dtypes in enumerate(ordered_datatypes):
    if a in dtypes and b in dtypes:
      return ordered_datatypes[idx + 1][0]
    if a in dtypes:
      return b
    if b in dtypes:
      return a

def _higher_type_kind(a: Optional[type], b: Optional[type]):
  ordered_type_kinds = (
    bool, int, float, complex
  )

  assert a is None or a in ordered_type_kinds, a
  assert b is None or b in ordered_type_kinds, b

  if a is b:
      return a

  if a is None:
    return b

  if b is None:
    return a

  for kind in ordered_type_kinds:
    if a is kind:
      return b
    if b is kind:
      return a

  raise ValueError("Received unknown types!")

# Returns the type promoted version of the original tensors and scalars.
# Type promotion works by first deciding which of four type kinds to use:
#   bool -> integer -> floating point -> complex
# The type kind is the highest type kind among all the arguments.
#
# Second the particular datatype in the type kind is determined.
# The datatypes are partially ordered as follows:
#
# bool -> uint8, int8 -> int16 -> int32 -> int64 ->
#   float16, bfloat16 -> float32 -> float64 -> complex32 -> complex64 -> complex128
#
# The particular datatype is chosen as follows:
#   - if no tensor is of the highest type kind, then the default datatype for that kind is chosen
#   - if there are only scalar tensors (tensors with no dimensions) of the highest type kind, then
#       the datatype is the highest scalar tensor datatype. If this is not uniquely determined then
#       it is the next highest datatype.
#   - if there are non-scalar tensors (tensors with one or more dimensions) of the highest type kind,
#       then the datatype is the highest non-scalar tensor datatype. If this is not uniquely determined
#       then it is the next highest datatype.
#
# Once the datatype is determined all tensors are cast to it and all scalars are cast to the corresponding
#   Python datatype (bool, int, float, complex).
# TODO: document int_to_float
# TODO: update to support changing default dtypes
def type_promote_elementwise(*_args, int_to_float=False):
  args = tuple(filter(lambda x: x is not None, _args))

  # Type checking
  for arg in args:
    assert isinstance(arg, (Number, Tensor))

  # Determines datatypes for each category
  scalar_args = filter(lambda x: isinstance(x, Number), args)
  scalar_type_kind = reduce(
    lambda acc, x: _higher_type_kind(type(acc), type(x)),
    scalar_args,
    False)

  scalar_tensors = filter(lambda t: isinstance(t, Tensor) and t.ndim == 0, args)
  scalar_tensor_dtype = reduce(
    _higher_dtype,
    (t.dtype for t in scalar_tensors),
    torch.bool
  )
  scalar_tensor_type_kind = _dtype_to_kind(scalar_tensor_dtype)

  nonscalar_tensors = filter(lambda t: isinstance(t, Tensor) and t.ndim != 0, args)
  nonscalar_tensor_dtype = reduce(
    _higher_dtype,
    (t.dtype for t in nonscalar_tensors),
    torch.bool
  )
  nonscalar_tensor_type_kind = _dtype_to_kind(nonscalar_tensor_dtype)

  type_kind = reduce(
    _higher_type_kind,
    (scalar_type_kind, scalar_tensor_type_kind, nonscalar_tensor_type_kind))

  if nonscalar_tensor_type_kind is type_kind:
    dtype = nonscalar_tensor_dtype
  elif scalar_tensor_type_kind is type_kind:
    dtype = scalar_tensor_dtype
  else:
    # scalar type kind -> default torch dtype mapping
    if type_kind is bool:
      dtype = torch.bool
    elif type_kind is int:
      dtype = torch.int64
    elif type_kind is float:
      dtype = torch.get_default_dtype()
    else:
      # type_kind is complex
      dtype = torch.complex128 if torch.get_default_dtype() is torch.float64 else torch.complex64

  # TODO: unconditionally insert prims and add a pass removing them
  def _maybe_convert_element_type(x, dtype):
    if x is None:
      return x

    if isinstance(x, Number):
      return _dtype_to_kind(dtype)(x)

    if isinstance(x, Tensor):
      if x.dtype is dtype:
        return x
      return prims.convert_element_type(x, dtype)

    raise NotImplementedError

  return map(lambda x: _maybe_convert_element_type(x, dtype), _args)

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
  common_shape = [1,] * reduce(max, (len(shape) for shape in shapes))
  for shape in shapes:
    for idx in range(-1, -1 - len(shape), -1):
      if common_shape[idx] == 1:
        if shape[idx] < 0:
          raise ValueError("Attempting to broadcast a dimension with negative length!")
        common_shape[idx] = shape[idx]
      elif shape[idx] != 1:
        assert common_shape[idx] == shape[idx]

  return common_shape

def broadcast(*args):
  # Computes common shape
  common_shape = _broadcast_shapes(*map(lambda t: t.shape if isinstance(t, Tensor) else None, args))
  common_rank = len(common_shape) + 1

  def _maybe_broadcast(x, shape):
    if x is None:
      return None
    elif isinstance(x, Number):
      return x
    elif isinstance(x, Tensor):
      start = common_rank - (len(x.shape) + 1)
      dims = tuple(range(start, len(x.shape) + start))

      # TODO: add a pass to remove unnecessary broadcast_in_dim calls
      return prims.broadcast_in_dim(x, common_shape, dims)
    else:
      assert False, "Unexpected type when broadcasting: " + str(type(x)) + "!"

  return map(lambda x: _maybe_broadcast(x, common_shape), args)

def add(a, b, *, alpha=None, out=None):
  '''
  Reference implementation of torch.add
  '''
  a, b, alpha = type_promote_elementwise(a, b, alpha, int_to_float=False)
  a, b, alpha = broadcast(a, b, alpha)

  if alpha is not None:
    result = prims.mul(b, alpha)

  result = prims.add(a, b)

  if out is not None:
    return prims.copy_to(out, result)

  return result

