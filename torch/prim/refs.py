from numbers import Number
from functools import reduce

import torch
from torch.prim.tracing import (
    TensorLikes,
)
import torch.prim.utils as utils
import torch.prim.prims as prims

# refs
# Operations built on prims

# Private helper function.
# Returns the higher of two torch datatypes a and b or, if the two
#   are not ordered relative to each other, the next
#   higher datatype
def _higher_dtype(a: torch.dtype, b: torch.dtype):
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

def _higher_type_kind(a: type, b: type):
    # Type checking
    assert isinstance(a, type)
    assert isinstance(b, type)

    if a is b:
        return a

    ordered_type_kinds = (
        bool, int, float, complex
    )

    for idx, kind in enumerate(ordered_type_kinds):
        if a is kind:
            return b
        if b is kind:
            return a

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
def type_promote_elementwise(*args, int_to_float=False):
    # Type checking
    for arg in args:
        assert isinstance(arg, (Number, TensorLikes))

    # Determines datatypes for each category
    scalar_args = filter(lambda x: isinstance(x, Number), args)
    scalar_type_kind = reduce(
        lambda x, y: _higher_type_kind(type(x), type(y)),
        scalar_args,
        bool)

    scalar_tensors = filter(lambda t: isinstance(t, TensorLikes) and t.ndim == 0, args)
    scalar_tensor_dtype = reduce(
        _higher_dtype,
        (t.dtype for t in scalar_tensors),
        torch.bool
    )
    scalar_tensor_type_kind = utils.dtype_to_kind(scalar_tensor_dtype)

    nonscalar_tensors = filter(lambda t: isinstance(t, TensorLikes) and t.ndim != 0, args)
    nonscalar_tensor_dtype = reduce(
        _higher_dtype,
        (t.dtype for t in nonscalar_tensors),
        torch.bool
    )
    nonscalar_tensor_type_kind = utils.dtype_to_kind(nonscalar_tensor_dtype)

    type_kind = reduce(
        _higher_type_kind,
        (scalar_type_kind, scalar_tensor_type_kind, nonscalar_tensor_type_kind)
    )

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

    def _maybe_convert_element_type(x, dtype):
        if isinstance(x, Number):
            return utils.dtype_to_kind(dtype)(x)

        if isinstance(x, TensorLikes):
            if x.dtype is dtype:
                return x
            return prims.convert_element_type(x, dtype)

        raise NotImplementedError

    return map(lambda x: _maybe_convert_element_type(x, dtype), args)

# TODO: working here
def broadcast(*args):
  # Computes common shape
  common_shape = utils.broadcast_shapes(*map(lambda t: t.shape, args))

  def _maybe_broadcast(x, shape):
    if isinstance(x, Number):
      return x
    elif isinstance(x, TensorLikes):
      if not utils.requires_broadcasting(x, common_shape):
        return x

      return prims.expand(x, common_shape)
    else:
      assert False, "Unexpected type when broadcasting: " + str(type(x)) + "!"

  return map(lambda x: _maybe_broadcast(x, common_shape), args)

# ref.add adds
def add(a, b):
  utils.check_same_device(a, b, allow_scalars=True)
  a, b = type_promote_elementwise(a, b, int_to_float=False)
  a, b = broadcast(a, b)
  return prims.add(a, b)