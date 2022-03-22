from numbers import Number
from typing import Sequence
from itertools import zip_longest
from functools import reduce

import torch
from torch.prim.tracing import (
    TensorLikes, DimProxy, ShapeProxy
)
import torch.prim.prims as prims

# prim.utils
# prim.utils contains basic operations that prims need but that never appear in traces

# Private helper function.
# Returns the Python type (AKA "type kind") corresponding to the torch dtype.
def dtype_to_kind(dtype):
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
def check_same_dtype(*args):
  full_dtype = None
  scalar_dtype = None

  for arg in args:
    if isinstance(arg, Number):
      if scalar_dtype is None:
        scalar_dtype = type(arg)

      assert scalar_dtype is type(arg), "Scalar of type " + str(type(arg)) + " is not the expected dtype of " + str(scalar_dtype) + "!"
    elif isinstance(arg, TensorLikes):
      if full_dtype is None:
        full_dtype = arg.dtype
      if scalar_dtype is None:
        scalar_dtype = dtype_to_kind(arg.dtype)

      assert full_dtype is arg.dtype, "Tensor with dtype " + str(arg.dtype) + " is not the expected dtype of " + str(full_dtype) + "!"
      assert dtype_to_kind(arg.dtype) is scalar_dtype, "Tensor with dtype " + str(arg.dtype) + " is not the expected dtype kind of " + str(scalar_dtype) + "!"
    else:
      assert False, "Unexpected type when checking for same dtype, " + str(type(arg)) + "!"

# TODO: document
def require_same_length(a, b):
  if isinstance(a, Number) and isinstance(b, Number):
    if a != b:
      raise ValueError("a ", str(a), " is not the same length as b ", str(b))
  else:
    assert isinstance(a, DimProxy)
    assert isinstance(b, DimProxy)
    a.require_same_length(b)

# TODO: document
def require_same_shape(a, b):
  # Type checks
  assert isinstance(a, Sequence)
  assert isinstance(b, Sequence)

  if len(a) != len(b):
    raise ValueError("Cannot require two shapes with different rank to be the same!")

  for x, y in zip(a, b):
    require_same_length(x, y)

# Asserts if any of the following are true:
#   - a non-scalar or non-Tensor is given
#   - the shape of any tensors is distinct
# TODO: change asserts to RuntimeError and ValueError exceptions
# TODO: update to handle containers
def check_same_shape(*args):
  shape = None

  for arg in args:
    if isinstance(arg, Number):
      continue
    elif isinstance(arg, TensorLikes):
      if shape is None:
        shape = arg.shape

      require_same_shape(shape, arg.shape)
    else:
      assert False, "Unexpected type when checking for same shape, " + str(type(arg)) + "!"

# Asserts if any of the following are true:
#   - a non-scalar or non-Tensor is given
#   - the shape of any tensors is distinct
# TODO: change asserts to RuntimeError and ValueError exceptions
# TODO: update to handle containers
def check_same_device(*args, allow_scalars):
  if len(args) == 0:
    return

  # Note: cannot initialize device to the first arg's device (it may not have one)
  device = None
  for arg in args:
    if isinstance(arg, Number):
      assert allow_scalars, "Found a scalar when checking for same device but scalars not allowed!"
    elif isinstance(arg, TensorLikes):
      if device is None:
        device = arg.device

      assert device == arg.device, "Tensor on device " + str(arg.device) + " is not on the expected device " + str(device) + "!"
    else:
      assert False, "Unexpected type when checking for same device, " + str(type(arg)) + "!"

# TODO: comment
def requires_broadcasting(t, shape):
  # Type checking
  assert isinstance(t, (TensorLikes, Sequence))
  assert isinstance(shape, Sequence)

  if isinstance(t, TensorLikes):
    base = t.shape
  else:
    base = t

  if len(shape) < len(base):
    raise ValueError("Attempting to expand a tensor to a lower rank!")

  # Short-circuits if it's trivial to verify an expansion occurred
  if len(shape) > len(base):
    return True

  # Verifies that an existing dimension is broadcast
  for idx in range(-1, -1 - len(base), -1):
    if base[idx] == 1 and shape[idx] != 1:
      return True

  return False


# def assert_shape_broadcast(lhs, rhs):
#     r = []
#     for x, y in itertools.zip_longest(
#         reversed(lhs.shape), reversed(rhs.shape), fillvalue=1
#     ):
#         if definitely_one(x):
#             r.append(y)
#         elif definitely_one(y):
#             r.append(x)
#         else:
#             assert_int_eq(x, y)
#             r.append(x)  # pick one arbitrarily
#     return tuple(reversed(r))

# Construction ops
# def make_shape(node, shape: Sequence):
#     # Type checks
#     assert isinstance(shape, Sequence)

#     if isinstance(shape, ShapeProxy):
#         return shape

#     # shape is a non-proxy sequence
#     for dim in shape:
#         if isinstance(dim, DimProxy):
#             ctx = dim.ctx
#             name = ctx.tensor_name()
#             return ShapeProxy(dim.ctx, node, name, shape)

#     # Fully-instantiated ("concrete") shape (a tuple of Python numbers)
#     return tuple(shape)

# TODO: document
def broadcast_shapes(*shapes):
    # Short-circuits on no input
    if len(shapes) == 0:
        return None

    # Type checking
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
                require_same_length(common_shape[idx], shape[idx])

    # Handles shape proxies
    for dim in common_shape:
      if isinstance(dim, DimProxy):
        ctx = dim.ctx
        output_name = ctx.shape_name()
        node = ctx.graph.create_node(
          'call_function',
          broadcast_shapes,
          name=output_name,
          args=shapes)
        for shape in shapes:
          shape.node.users[node] = None
        return ShapeProxy(ctx, node, output_name, common_shape)

    return common_shape

# Checks for same device and dtype
def elementwise_binary_checks(a, b):
    check_same_device(a, b, allow_scalars=True)
    check_same_dtype(a, b)
    check_same_shape(a, b)

# TODO: reconsider whether this function should exist -- probably not?
# def prim_enumerate_common_dims(*args):
#   if len(args) == 0:
#     return None

#   min_len = len(args[0].shape) if isinstance(args[0], TensorLikes) else len(args[0])
#   for arg in args:
#     if isinstance(arg, TensorLikes):
#       min_len = min(len(arg.shape), min_len)
#     elif isinstance(arg, Sequence):
#       min_len = min(len(arg), min_len)
#     else:
#       assert False

# Compares the length of two dimensions.
# Returns:
# 1 if a > b
# 0 if a == b
# -1 if a < b
# None if a cannot be compared to b (because both are symbols of unknown length)
# TODO: review this
# def prim_utils_compare_length(a, b):
#   if isinstance(a, Number) and isinstance(b, Number):
#     if a > b:
#       return 1
#     elif a == b:
#       return 0
#     # a < b
#     return -1
#   # TODO: add symbol logic
#   raise NotImplementedError

# prim.utils.compare_length = prim_utils_compare_length



