
/**
 * @generated
 * This is an auto-generated file. Please do not modify it by hand.
 * To re-generate, please run:
 * cd ~/pytorch && python
 * torchgen/shape_functions/gen_jit_shape_functions.py
 */
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/serialized_shape_function_registry.h>

// clang-format off

namespace torch::jit {


std::string shape_funcs = ""
+ std::string(R"=====(
def unary(self: List[int]) -> List[int]:
  out = annotate(List[int], [])
  for _0 in range(torch.len(self)):
    elem = self[_0]
    _1 = torch.append(out, elem)
  return out

def adaptive_avg_pool2d(self: List[int],
    out: List[int]) -> List[int]:
  if torch.eq(torch.len(out), 2):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(self), 3):
    _0 = True
  else:
    _0 = torch.eq(torch.len(self), 4)
  if _0:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _1 = torch.__range_length(1, torch.len(self), 1)
  for _2 in range(_1):
    i = torch.__derive_index(_2, 1, 1)
    if torch.ne(self[i], 0):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  shape = annotate(List[int], [])
  _3 = torch.__range_length(0, torch.sub(torch.len(self), 2), 1)
  for _4 in range(_3):
    i0 = torch.__derive_index(_4, 0, 1)
    _5 = torch.append(shape, self[i0])
  for _6 in range(torch.len(out)):
    elem = out[_6]
    _7 = torch.append(shape, elem)
  return shape

def zero_dim_tensor(input: Any) -> List[int]:
  return annotate(List[int], [])

def arange_end(end: Union[float, int],
    inp0: Any,
    inp1: Any,
    inp2: Any,
    inp3: Any) -> List[int]:
  if torch.ge(end, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  return [int(torch.ceil(end))]

def arange_start(start: Union[float, int],
    end: Union[float, int],
    inp0: Any,
    inp1: Any,
    inp2: Any,
    inp3: Any) -> List[int]:
  if torch.ge(end, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(end, start):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _0 = int(torch.ceil(torch.sub(end, start)))
  return [_0]

)=====")
+ std::string(R"=====(def arange_start_step(start: Union[float, int],
    end: Union[float, int],
    step: Union[float, int],
    inp0: Any,
    inp1: Any,
    inp2: Any,
    inp3: Any) -> List[int]:
  if torch.ne(step, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(step, 0):
    if torch.ge(start, end):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  else:
    if torch.ge(end, start):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  _0 = torch.div(torch.sub(end, start), step)
  return [torch.ceil(_0)]

def squeeze_nodim(li: List[int]) -> List[int]:
  out = annotate(List[int], [])
  for i in range(torch.len(li)):
    if torch.ne(li[i], 1):
      _0 = torch.append(out, li[i])
    else:
      pass
  return out

def squeeze(li: List[int],
    dim: int) -> List[int]:
  out = annotate(List[int], [])
  _0 = torch.len(li)
  if torch.le(_0, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = _0
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(dim, min):
    _1 = True
  else:
    _1 = torch.gt(dim, max)
  if torch.__not__(_1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim, 0):
    wrapped_dim = torch.add(dim, dim_post_expr)
  else:
    wrapped_dim = dim
  for i in range(torch.len(li)):
    if torch.eq(i, wrapped_dim):
      if torch.ne(li[i], 1):
        _2 = torch.append(out, li[i])
      else:
        pass
    else:
      _3 = torch.append(out, li[i])
  return out

)=====")
+ std::string(R"=====(def squeeze_dims(li: List[int],
    dims: List[int]) -> List[int]:
  if torch.eq(torch.len(dims), 0):
    _0 = li
  else:
    wrapped_dims = annotate(List[int], [])
    for _1 in range(torch.len(dims)):
      elem = dims[_1]
      _2 = torch.append(wrapped_dims, elem)
    for i in range(torch.len(dims)):
      _3 = wrapped_dims[i]
      _4 = torch.len(li)
      if torch.le(_4, 0):
        dim_post_expr = 1
      else:
        dim_post_expr = _4
      min = torch.neg(dim_post_expr)
      max = torch.sub(dim_post_expr, 1)
      if torch.lt(_3, min):
        _5 = True
      else:
        _5 = torch.gt(_3, max)
      if torch.__not__(_5):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      if torch.lt(_3, 0):
        dim = torch.add(_3, dim_post_expr)
      else:
        dim = _3
      _6 = torch._set_item(wrapped_dims, i, dim)
    result = annotate(List[int], [])
    for i0 in range(torch.len(li)):
      if torch.eq(li[i0], 1):
        _7 = torch.__contains__(wrapped_dims, i0)
        if torch.__not__(_7):
          _8 = torch.append(result, li[i0])
        else:
          pass
      else:
        _9 = torch.append(result, li[i0])
    _0 = result
  return _0

def unsqueeze(li: List[int],
    dim: int) -> List[int]:
  _0 = torch.add(torch.len(li), 1)
  if torch.le(_0, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = _0
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(dim, min):
    _1 = True
  else:
    _1 = torch.gt(dim, max)
  if torch.__not__(_1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim, 0):
    dim0 = torch.add(dim, dim_post_expr)
  else:
    dim0 = dim
  out = annotate(List[int], [])
  for _2 in range(torch.len(li)):
    elem = li[_2]
    _3 = torch.append(out, elem)
  torch.insert(out, dim0, 1)
  return out

)=====")
+ std::string(R"=====(def slice(self: List[int],
    dim: int,
    start: Optional[int],
    end: Optional[int],
    step: int) -> List[int]:
  ndim = torch.len(self)
  if torch.ne(ndim, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.le(ndim, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = ndim
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(dim, min):
    _0 = True
  else:
    _0 = torch.gt(dim, max)
  if torch.__not__(_0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim, 0):
    dim0 = torch.add(dim, dim_post_expr)
  else:
    dim0 = dim
  if torch.__isnot__(start, None):
    start_val = unchecked_cast(int, start)
  else:
    start_val = 0
  if torch.__isnot__(end, None):
    end_val = unchecked_cast(int, end)
  else:
    end_val = 9223372036854775807
  if torch.gt(step, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _1 = torch.eq(start_val, 9223372036854775807)
  if _1:
    start_val0 = 0
  else:
    start_val0 = start_val
  if torch.lt(start_val0, 0):
    start_val1 = torch.add(start_val0, self[dim0])
  else:
    start_val1 = start_val0
  if torch.lt(end_val, 0):
    end_val0 = torch.add(end_val, self[dim0])
  else:
    end_val0 = end_val
  if torch.lt(start_val1, 0):
    start_val2 = 0
  else:
    if torch.gt(start_val1, self[dim0]):
      start_val3 = self[dim0]
    else:
      start_val3 = start_val1
    start_val2 = start_val3
  if torch.lt(end_val0, start_val2):
    end_val1 = start_val2
  else:
    if torch.ge(end_val0, self[dim0]):
      end_val2 = self[dim0]
    else:
      end_val2 = end_val0
    end_val1 = end_val2
  slice_len = torch.sub(end_val1, start_val2)
  out = annotate(List[int], [])
  for _2 in range(torch.len(self)):
    elem = self[_2]
    _3 = torch.append(out, elem)
  _4 = torch.sub(torch.add(slice_len, step), 1)
  _5 = torch._set_item(out, dim0, torch.floordiv(_4, step))
  return out

)=====")
+ std::string(R"=====(def select(self: List[int],
    dim: int,
    index: int) -> List[int]:
  ndim = torch.len(self)
  if torch.ne(ndim, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.le(ndim, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = ndim
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(dim, min):
    _0 = True
  else:
    _0 = torch.gt(dim, max)
  if torch.__not__(_0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim, 0):
    dim0 = torch.add(dim, dim_post_expr)
  else:
    dim0 = dim
  size = self[dim0]
  if torch.lt(index, torch.neg(size)):
    _1 = True
  else:
    _1 = torch.ge(index, size)
  if torch.__not__(_1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  out = annotate(List[int], [])
  for i in range(ndim):
    if torch.ne(i, dim0):
      _2 = torch.append(out, self[i])
    else:
      pass
  return out

)=====")
+ std::string(R"=====(def index_select(self: List[int],
    dim: int,
    index: List[int]) -> List[int]:
  _0 = torch.len(self)
  if torch.le(_0, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = _0
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(dim, min):
    _1 = True
  else:
    _1 = torch.gt(dim, max)
  if torch.__not__(_1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim, 0):
    dim0 = torch.add(dim, dim_post_expr)
  else:
    dim0 = dim
  numel = 1
  for _2 in range(torch.len(index)):
    elem = index[_2]
    numel = torch.mul(numel, elem)
  if torch.le(torch.len(index), 1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(dim0, 0):
    _3 = True
  else:
    _3 = torch.lt(dim0, torch.len(self))
  if _3:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  result_size = annotate(List[int], [])
  for i in range(torch.len(self)):
    if torch.eq(dim0, i):
      _4 = torch.append(result_size, numel)
    else:
      _5 = torch.append(result_size, self[i])
  return result_size

)=====")
+ std::string(R"=====(def embedding(weight: List[int],
    indices: List[int],
    padding_idx: int=-1,
    scale_grad_by_freq: bool=False,
    sparse: bool=False) -> List[int]:
  if torch.eq(torch.len(weight), 2):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(indices), 1):
    _1 = torch.len(weight)
    if torch.le(_1, 0):
      dim_post_expr = 1
    else:
      dim_post_expr = _1
    min = torch.neg(dim_post_expr)
    max = torch.sub(dim_post_expr, 1)
    if torch.lt(0, min):
      _2 = True
    else:
      _2 = torch.gt(0, max)
    if torch.__not__(_2):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    numel = 1
    for _3 in range(torch.len(indices)):
      elem = indices[_3]
      numel = torch.mul(numel, elem)
    if torch.le(torch.len(indices), 1):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    result_size = annotate(List[int], [])
    for i in range(torch.len(weight)):
      if torch.eq(0, i):
        _4 = torch.append(result_size, numel)
      else:
        _5 = torch.append(result_size, weight[i])
    _0 = result_size
  else:
    size = annotate(List[int], [])
    for _6 in range(torch.len(indices)):
      elem0 = indices[_6]
      _7 = torch.append(size, elem0)
    _8 = torch.append(size, weight[1])
    _0 = size
  return _0

def mm(self: List[int],
    mat2: List[int]) -> List[int]:
  _0 = "AssertionError: self must be a matrix"
  _1 = "AssertionError: mat2 must be a matrix"
  if torch.eq(torch.len(self), 2):
    pass
  else:
    ops.prim.RaiseException(_0)
  if torch.eq(torch.len(mat2), 2):
    pass
  else:
    ops.prim.RaiseException(_1)
  if torch.eq(self[1], mat2[0]):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  return [self[0], mat2[1]]

)=====")
+ std::string(R"=====(def dot(self: List[int],
    tensor: List[int]) -> List[int]:
  if torch.eq(torch.len(self), 1):
    _0 = torch.eq(torch.len(tensor), 1)
  else:
    _0 = False
  if _0:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(self[0], tensor[0]):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  return annotate(List[int], [])

def mv(self: List[int],
    vec: List[int]) -> List[int]:
  if torch.eq(torch.len(self), 2):
    _0 = torch.eq(torch.len(vec), 1)
  else:
    _0 = False
  if _0:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(self[1], vec[0]):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  return [self[0]]

)=====")
+ std::string(R"=====(def matmul(tensor1: List[int],
    tensor2: List[int]) -> List[int]:
  _0 = "AssertionError: self must be a matrix"
  _1 = "AssertionError: mat2 must be a matrix"
  _2 = "AssertionError: self must be a matrix"
  _3 = "AssertionError: mat2 must be a matrix"
  _4 = "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
  _5 = "AssertionError: both  arguments to matmul need to be at least 1D"
  _6 = uninitialized(List[int])
  dim_tensor1 = torch.len(tensor1)
  dim_tensor2 = torch.len(tensor2)
  if torch.eq(dim_tensor1, 1):
    _7 = torch.eq(dim_tensor2, 1)
  else:
    _7 = False
  if _7:
    if torch.eq(torch.len(tensor1), 1):
      _9 = torch.eq(torch.len(tensor2), 1)
    else:
      _9 = False
    if _9:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    if torch.eq(tensor1[0], tensor2[0]):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    _8 = annotate(List[int], [])
  else:
    if torch.eq(dim_tensor1, 2):
      _10 = torch.eq(dim_tensor2, 1)
    else:
      _10 = False
    if _10:
      if torch.eq(torch.len(tensor1), 2):
        _12 = torch.eq(torch.len(tensor2), 1)
      else:
        _12 = False
      if _12:
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      if torch.eq(tensor1[1], tensor2[0]):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      _11 = [tensor1[0]]
    else:
      if torch.eq(dim_tensor1, 1):
        _13 = torch.eq(dim_tensor2, 2)
      else:
        _13 = False
      if _13:
        _15 = torch.add(torch.len(tensor1), 1)
        if torch.le(_15, 0):
          dim_post_expr = 1
        else:
          dim_post_expr = _15
        min = torch.neg(dim_post_expr)
        max = torch.sub(dim_post_expr, 1)
        if torch.lt(0, min):
          _16 = True
        else:
          _16 = torch.gt(0, max)
        if torch.__not__(_16):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        out = annotate(List[int], [])
        for _17 in range(torch.len(tensor1)):
          elem = tensor1[_17]
          _18 = torch.append(out, elem)
        torch.insert(out, 0, 1)
        if torch.eq(torch.len(out), 2):
          pass
        else:
          ops.prim.RaiseException(_0)
        if torch.eq(torch.len(tensor2), 2):
          pass
        else:
          ops.prim.RaiseException(_1)
        if torch.eq(out[1], tensor2[0]):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        _19 = [out[0], tensor2[1]]
        out0 = annotate(List[int], [])
        for i in range(2):
          if torch.eq(i, 0):
            if torch.ne(_19[i], 1):
              _20 = torch.append(out0, _19[i])
            else:
              pass
          else:
            _21 = torch.append(out0, _19[i])
        _14 = out0
      else:
        if torch.eq(dim_tensor1, 2):
          _22 = torch.eq(dim_tensor2, 2)
        else:
          _22 = False
        if _22:
          _24 = torch.eq(torch.len(tensor1), 2)
          if _24:
            pass
          else:
            ops.prim.RaiseException(_2)
          _25 = torch.eq(torch.len(tensor2), 2)
          if _25:
            pass
          else:
            ops.prim.RaiseException(_3)
          _26 = torch.eq(tensor1[1], tensor2[0])
          if _26:
            pass
          else:
            ops.prim.RaiseException("AssertionError: ")
          _23 = [tensor1[0], tensor2[1]]
        else:
          if torch.ge(dim_tensor1, 1):
            _27 = torch.ge(dim_tensor2, 1)
          else:
            _27 = False
          if _27:
            if torch.gt(dim_tensor1, 1):
              n = tensor1[-2]
            else:
              n = 1
            batch_tensor1 = annotate(List[int], [])
            for i0 in range(torch.sub(dim_tensor1, 2)):
              _29 = torch.append(batch_tensor1, tensor1[i0])
            p = tensor2[-1]
            batch_tensor2 = annotate(List[int], [])
            for i1 in range(torch.sub(dim_tensor2, 2)):
              _30 = torch.append(batch_tensor2, tensor2[i1])
            dimsA = torch.len(batch_tensor1)
            dimsB = torch.len(batch_tensor2)
            ndim = ops.prim.max(dimsA, dimsB)
            expand_batch_portion = annotate(List[int], [])
            for i2 in range(ndim):
              offset = torch.sub(torch.sub(ndim, 1), i2)
              dimA = torch.sub(torch.sub(dimsA, 1), offset)
              dimB = torch.sub(torch.sub(dimsB, 1), offset)
              if torch.ge(dimA, 0):
                sizeA = batch_tensor1[dimA]
              else:
                sizeA = 1
              if torch.ge(dimB, 0):
                sizeB = batch_tensor2[dimB]
              else:
                sizeB = 1
              if torch.ne(sizeA, sizeB):
                _31 = torch.ne(sizeA, 1)
              else:
                _31 = False
              if _31:
                _32 = torch.ne(sizeB, 1)
              else:
                _32 = False
              if _32:
                _33 = torch.format(_4, sizeA, sizeB, i2)
                _34 = torch.add("AssertionError: ", _33)
                ops.prim.RaiseException(_34)
              else:
                pass
              if torch.eq(sizeA, 1):
                _35 = sizeB
              else:
                _35 = sizeA
              _36 = torch.append(expand_batch_portion, _35)
            if torch.gt(dim_tensor1, 1):
              _37 = torch.append(expand_batch_portion, n)
            else:
              pass
            if torch.gt(dim_tensor2, 1):
              _38 = torch.append(expand_batch_portion, p)
            else:
              pass
            _28 = expand_batch_portion
          else:
            ops.prim.RaiseException(_5)
            _28 = _6
          _23 = _28
        _14 = _23
      _11 = _14
    _8 = _11
  return _8

)=====")
+ std::string(R"=====(def linear(input: List[int],
    weight: List[int],
    bias: Optional[List[int]]) -> List[int]:
  _0 = "AssertionError: self must be a matrix"
  _1 = "AssertionError: mat2 must be a matrix"
  _2 = "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
  _3 = "AssertionError: both  arguments to matmul need to be at least 1D"
  _4 = "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
  if torch.le(torch.len(weight), 2):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  self_len = torch.len(weight)
  if torch.eq(self_len, 0):
    _5 = annotate(List[int], [])
  else:
    if torch.eq(self_len, 1):
      _6 = [weight[0]]
    else:
      _6 = [weight[1], weight[0]]
    _5 = _6
  _7 = uninitialized(List[int])
  dim_tensor1 = torch.len(input)
  dim_tensor2 = torch.len(_5)
  if torch.eq(dim_tensor1, 1):
    _8 = torch.eq(dim_tensor2, 1)
  else:
    _8 = False
  if _8:
    if torch.eq(torch.len(input), 1):
      _9 = torch.eq(torch.len(_5), 1)
    else:
      _9 = False
    if _9:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    if torch.eq(input[0], _5[0]):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    out = annotate(List[int], [])
  else:
    if torch.eq(dim_tensor1, 2):
      _10 = torch.eq(dim_tensor2, 1)
    else:
      _10 = False
    if _10:
      if torch.eq(torch.len(input), 2):
        _12 = torch.eq(torch.len(_5), 1)
      else:
        _12 = False
      if _12:
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      if torch.eq(input[1], _5[0]):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      _11 = [input[0]]
    else:
      if torch.eq(dim_tensor1, 1):
        _13 = torch.eq(dim_tensor2, 2)
      else:
        _13 = False
      if _13:
        _15 = torch.add(torch.len(input), 1)
        if torch.le(_15, 0):
          dim_post_expr = 1
        else:
          dim_post_expr = _15
        min = torch.neg(dim_post_expr)
        max = torch.sub(dim_post_expr, 1)
        if torch.lt(0, min):
          _16 = True
        else:
          _16 = torch.gt(0, max)
        if torch.__not__(_16):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        out0 = annotate(List[int], [])
        for _17 in range(torch.len(input)):
          elem = input[_17]
          _18 = torch.append(out0, elem)
        torch.insert(out0, 0, 1)
        if torch.eq(torch.len(out0), 2):
          pass
        else:
          ops.prim.RaiseException(_0)
        if torch.eq(torch.len(_5), 2):
          pass
        else:
          ops.prim.RaiseException(_1)
        if torch.eq(out0[1], _5[0]):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        _19 = [out0[0], _5[1]]
        out1 = annotate(List[int], [])
        for i in range(2):
          if torch.eq(i, 0):
            if torch.ne(_19[i], 1):
              _20 = torch.append(out1, _19[i])
            else:
              pass
          else:
            _21 = torch.append(out1, _19[i])
        _14 = out1
      else:
        if torch.eq(dim_tensor1, 2):
          _22 = torch.eq(dim_tensor2, 2)
        else:
          _22 = False
        if _22:
          if torch.eq(torch.len(input), 2):
            pass
          else:
            ops.prim.RaiseException(_0)
          if torch.eq(torch.len(_5), 2):
            pass
          else:
            ops.prim.RaiseException(_1)
          if torch.eq(input[1], _5[0]):
            pass
          else:
            ops.prim.RaiseException("AssertionError: ")
          _23 = [input[0], _5[1]]
        else:
          if torch.ge(dim_tensor1, 1):
            _24 = torch.ge(dim_tensor2, 1)
          else:
            _24 = False
          if _24:
            if torch.gt(dim_tensor1, 1):
              n = input[-2]
            else:
              n = 1
            batch_tensor1 = annotate(List[int], [])
            for i0 in range(torch.sub(dim_tensor1, 2)):
              _26 = torch.append(batch_tensor1, input[i0])
            p = _5[-1]
            batch_tensor2 = annotate(List[int], [])
            for i1 in range(torch.sub(dim_tensor2, 2)):
              _27 = torch.append(batch_tensor2, _5[i1])
            dimsA = torch.len(batch_tensor1)
            dimsB = torch.len(batch_tensor2)
            ndim = ops.prim.max(dimsA, dimsB)
            expand_batch_portion = annotate(List[int], [])
            for i2 in range(ndim):
              offset = torch.sub(torch.sub(ndim, 1), i2)
              dimA = torch.sub(torch.sub(dimsA, 1), offset)
              dimB = torch.sub(torch.sub(dimsB, 1), offset)
              if torch.ge(dimA, 0):
                sizeA = batch_tensor1[dimA]
              else:
                sizeA = 1
              if torch.ge(dimB, 0):
                sizeB = batch_tensor2[dimB]
              else:
                sizeB = 1
              if torch.ne(sizeA, sizeB):
                _28 = torch.ne(sizeA, 1)
              else:
                _28 = False
              if _28:
                _29 = torch.ne(sizeB, 1)
              else:
                _29 = False
              if _29:
                _30 = torch.format(_2, sizeA, sizeB, i2)
                _31 = torch.add("AssertionError: ", _30)
                ops.prim.RaiseException(_31)
              else:
                pass
              if torch.eq(sizeA, 1):
                _32 = sizeB
              else:
                _32 = sizeA
              _33 = torch.append(expand_batch_portion, _32)
            if torch.gt(dim_tensor1, 1):
              _34 = torch.append(expand_batch_portion, n)
            else:
              pass
            if torch.gt(dim_tensor2, 1):
              _35 = torch.append(expand_batch_portion, p)
            else:
              pass
            _25 = expand_batch_portion
          else:
            ops.prim.RaiseException(_3)
            _25 = _7
          _23 = _25
        _14 = _23
      _11 = _14
    out = _11
  if torch.__isnot__(bias, None):
    bias0 = unchecked_cast(List[int], bias)
    dimsA0 = torch.len(bias0)
    dimsB0 = torch.len(out)
    ndim0 = ops.prim.max(dimsA0, dimsB0)
    expandedSizes = annotate(List[int], [])
    for i3 in range(ndim0):
      offset0 = torch.sub(torch.sub(ndim0, 1), i3)
      dimA0 = torch.sub(torch.sub(dimsA0, 1), offset0)
      dimB0 = torch.sub(torch.sub(dimsB0, 1), offset0)
      if torch.ge(dimA0, 0):
        sizeA0 = bias0[dimA0]
      else:
        sizeA0 = 1
      if torch.ge(dimB0, 0):
        sizeB0 = out[dimB0]
      else:
        sizeB0 = 1
      if torch.ne(sizeA0, sizeB0):
        _36 = torch.ne(sizeA0, 1)
      else:
        _36 = False
      if _36:
        _37 = torch.ne(sizeB0, 1)
      else:
        _37 = False
      if _37:
        _38 = torch.format(_4, sizeA0, sizeB0, i3)
        _39 = torch.add("AssertionError: ", _38)
        ops.prim.RaiseException(_39)
      else:
        pass
      if torch.eq(sizeA0, 1):
        _40 = sizeB0
      else:
        _40 = sizeA0
      _41 = torch.append(expandedSizes, _40)
    if torch.eq(expandedSizes, out):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  else:
    pass
  return out

)=====")
+ std::string(R"=====(def max_pool2d(input: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    ceil_mode: bool) -> List[int]:
  _0 = "AssertionError: max_pool2d: kernel_size must either be a single int, or a tuple of two ints"
  _1 = "AssertionError: max_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
  _2 = "AssertionError: max_pool2d: padding must either be a single int, or a tuple of two ints"
  _3 = "AssertionError: max_pool2d: dilation must be either a single int, or a tuple of two ints"
  _4 = "AssertionError: stride should not be zeero"
  _5 = "AssertionError: stride should not be zeero"
  if torch.eq(torch.len(kernel_size), 1):
    _6 = True
  else:
    _6 = torch.eq(torch.len(kernel_size), 2)
  if _6:
    pass
  else:
    ops.prim.RaiseException(_0)
  kH = kernel_size[0]
  if torch.eq(torch.len(kernel_size), 1):
    kW = kH
  else:
    kW = kernel_size[1]
  if torch.eq(torch.len(stride), 0):
    _7 = True
  else:
    _7 = torch.eq(torch.len(stride), 1)
  if _7:
    _8 = True
  else:
    _8 = torch.eq(torch.len(stride), 2)
  if _8:
    pass
  else:
    ops.prim.RaiseException(_1)
  if torch.eq(torch.len(stride), 0):
    dH = kH
  else:
    dH = stride[0]
  if torch.eq(torch.len(stride), 0):
    dW = kW
  else:
    if torch.eq(torch.len(stride), 1):
      dW0 = dH
    else:
      dW0 = stride[1]
    dW = dW0
  if torch.eq(torch.len(padding), 1):
    _9 = True
  else:
    _9 = torch.eq(torch.len(padding), 2)
  if _9:
    pass
  else:
    ops.prim.RaiseException(_2)
  padH = padding[0]
  if torch.eq(torch.len(padding), 1):
    padW = padH
  else:
    padW = padding[1]
  if torch.eq(torch.len(dilation), 1):
    _10 = True
  else:
    _10 = torch.eq(torch.len(dilation), 2)
  if _10:
    pass
  else:
    ops.prim.RaiseException(_3)
  dilationH = dilation[0]
  if torch.eq(torch.len(dilation), 1):
    dilationW = dilationH
  else:
    dilationW = dilation[1]
  if torch.eq(torch.len(input), 3):
    _11 = True
  else:
    _11 = torch.eq(torch.len(input), 4)
  if _11:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 4):
    nbatch = input[-4]
  else:
    nbatch = 1
  nInputPlane = input[-3]
  inputHeight = input[-2]
  inputWidth = input[-1]
  if torch.ne(dH, 0):
    pass
  else:
    ops.prim.RaiseException(_4)
  _12 = torch.add(torch.add(inputHeight, padH), padH)
  _13 = torch.mul(dilationH, torch.sub(kH, 1))
  _14 = torch.sub(torch.sub(_12, _13), 1)
  if ceil_mode:
    _15 = torch.sub(dH, 1)
  else:
    _15 = 0
  _16 = torch.floordiv(torch.add(_14, _15), dH)
  outputSize = torch.add(_16, 1)
  if ceil_mode:
    _17 = torch.ge(torch.mul(_16, dH), torch.add(inputHeight, padH))
    if _17:
      outputSize0 = _16
    else:
      outputSize0 = outputSize
    outputHeight = outputSize0
  else:
    outputHeight = outputSize
  if torch.ne(dW, 0):
    pass
  else:
    ops.prim.RaiseException(_5)
  _18 = torch.add(torch.add(inputWidth, padW), padW)
  _19 = torch.mul(dilationW, torch.sub(kW, 1))
  _20 = torch.sub(torch.sub(_18, _19), 1)
  if ceil_mode:
    _21 = torch.sub(dW, 1)
  else:
    _21 = 0
  _22 = torch.floordiv(torch.add(_20, _21), dW)
  outputSize1 = torch.add(_22, 1)
  if ceil_mode:
    _23 = torch.ge(torch.mul(_22, dW), torch.add(inputWidth, padW))
    if _23:
      outputSize2 = _22
    else:
      outputSize2 = outputSize1
    outputWidth = outputSize2
  else:
    outputWidth = outputSize1
  ndim = torch.len(input)
  if torch.gt(kW, 0):
    _24 = torch.gt(kH, 0)
  else:
    _24 = False
  if _24:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.gt(dW, 0):
    _25 = torch.gt(dH, 0)
  else:
    _25 = False
  if _25:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.gt(dilationH, 0):
    _26 = torch.gt(dilationW, 0)
  else:
    _26 = False
  if _26:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ne(input[1], 0):
    valid_dims = torch.ne(input[2], 0)
  else:
    valid_dims = False
  if torch.eq(ndim, 3):
    _27 = torch.ne(input[0], 0)
  else:
    _27 = False
  if _27:
    _28 = valid_dims
  else:
    _28 = False
  if _28:
    _29 = True
  else:
    if torch.eq(ndim, 4):
      _30 = valid_dims
    else:
      _30 = False
    if _30:
      _31 = torch.ne(input[3], 0)
    else:
      _31 = False
    _29 = _31
  if _29:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(torch.floordiv(kW, 2), padW):
    _33 = torch.ge(torch.floordiv(kH, 2), padH)
    _32 = _33
  else:
    _32 = False
  if _32:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(outputWidth, 1):
    _34 = torch.ge(outputHeight, 1)
  else:
    _34 = False
  if _34:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 3):
    _36 = [nInputPlane, outputHeight, outputWidth]
    _35 = _36
  else:
    _37 = [nbatch, nInputPlane, outputHeight, outputWidth]
    _35 = _37
  return _35

)=====")
+ std::string(R"=====(def max_pool2d_with_indices(input: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    ceil_mode: bool) -> Tuple[List[int], List[int]]:
  _0 = "AssertionError: max_pool2d: kernel_size must either be a single int, or a tuple of two ints"
  _1 = "AssertionError: max_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
  _2 = "AssertionError: max_pool2d: padding must either be a single int, or a tuple of two ints"
  _3 = "AssertionError: max_pool2d: dilation must be either a single int, or a tuple of two ints"
  _4 = "AssertionError: stride should not be zeero"
  if torch.eq(torch.len(kernel_size), 1):
    _5 = True
  else:
    _5 = torch.eq(torch.len(kernel_size), 2)
  if _5:
    pass
  else:
    ops.prim.RaiseException(_0)
  kH = kernel_size[0]
  if torch.eq(torch.len(kernel_size), 1):
    kW = kH
  else:
    kW = kernel_size[1]
  if torch.eq(torch.len(stride), 0):
    _6 = True
  else:
    _6 = torch.eq(torch.len(stride), 1)
  if _6:
    _7 = True
  else:
    _7 = torch.eq(torch.len(stride), 2)
  if _7:
    pass
  else:
    ops.prim.RaiseException(_1)
  if torch.eq(torch.len(stride), 0):
    dH = kH
  else:
    dH = stride[0]
  if torch.eq(torch.len(stride), 0):
    dW = kW
  else:
    if torch.eq(torch.len(stride), 1):
      dW0 = dH
    else:
      dW0 = stride[1]
    dW = dW0
  if torch.eq(torch.len(padding), 1):
    _8 = True
  else:
    _8 = torch.eq(torch.len(padding), 2)
  if _8:
    pass
  else:
    ops.prim.RaiseException(_2)
  padH = padding[0]
  if torch.eq(torch.len(padding), 1):
    padW = padH
  else:
    padW = padding[1]
  if torch.eq(torch.len(dilation), 1):
    _9 = True
  else:
    _9 = torch.eq(torch.len(dilation), 2)
  if _9:
    pass
  else:
    ops.prim.RaiseException(_3)
  dilationH = dilation[0]
  if torch.eq(torch.len(dilation), 1):
    dilationW = dilationH
  else:
    dilationW = dilation[1]
  if torch.eq(torch.len(input), 3):
    _10 = True
  else:
    _10 = torch.eq(torch.len(input), 4)
  if _10:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 4):
    nbatch = input[-4]
  else:
    nbatch = 1
  nInputPlane = input[-3]
  inputHeight = input[-2]
  inputWidth = input[-1]
  if torch.ne(dH, 0):
    pass
  else:
    ops.prim.RaiseException(_4)
  _11 = torch.add(torch.add(inputHeight, padH), padH)
  _12 = torch.mul(dilationH, torch.sub(kH, 1))
  _13 = torch.sub(torch.sub(_11, _12), 1)
  if ceil_mode:
    _14 = torch.sub(dH, 1)
  else:
    _14 = 0
  _15 = torch.floordiv(torch.add(_13, _14), dH)
  outputSize = torch.add(_15, 1)
  if ceil_mode:
    _16 = torch.ge(torch.mul(_15, dH), torch.add(inputHeight, padH))
    if _16:
      outputSize0 = _15
    else:
      outputSize0 = outputSize
    outputHeight = outputSize0
  else:
    outputHeight = outputSize
  if torch.ne(dW, 0):
    pass
  else:
    ops.prim.RaiseException(_4)
  _17 = torch.add(torch.add(inputWidth, padW), padW)
  _18 = torch.mul(dilationW, torch.sub(kW, 1))
  _19 = torch.sub(torch.sub(_17, _18), 1)
  if ceil_mode:
    _20 = torch.sub(dW, 1)
  else:
    _20 = 0
  _21 = torch.floordiv(torch.add(_19, _20), dW)
  outputSize1 = torch.add(_21, 1)
  if ceil_mode:
    _22 = torch.ge(torch.mul(_21, dW), torch.add(inputWidth, padW))
    if _22:
      outputSize2 = _21
    else:
      outputSize2 = outputSize1
    outputWidth = outputSize2
  else:
    outputWidth = outputSize1
  ndim = torch.len(input)
  if torch.gt(kW, 0):
    _23 = torch.gt(kH, 0)
  else:
    _23 = False
  if _23:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.gt(dW, 0):
    _24 = torch.gt(dH, 0)
  else:
    _24 = False
  if _24:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.gt(dilationH, 0):
    _25 = torch.gt(dilationW, 0)
  else:
    _25 = False
  if _25:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ne(input[1], 0):
    valid_dims = torch.ne(input[2], 0)
  else:
    valid_dims = False
  if torch.eq(ndim, 3):
    _26 = torch.ne(input[0], 0)
  else:
    _26 = False
  if _26:
    _27 = valid_dims
  else:
    _27 = False
  if _27:
    _28 = True
  else:
    if torch.eq(ndim, 4):
      _29 = valid_dims
    else:
      _29 = False
    if _29:
      _30 = torch.ne(input[3], 0)
    else:
      _30 = False
    _28 = _30
  if _28:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(torch.floordiv(kW, 2), padW):
    _32 = torch.ge(torch.floordiv(kH, 2), padH)
    _31 = _32
  else:
    _31 = False
  if _31:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(outputWidth, 1):
    _33 = torch.ge(outputHeight, 1)
  else:
    _33 = False
  if _33:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 3):
    _34 = [nInputPlane, outputHeight, outputWidth]
    out = _34
  else:
    _35 = [nbatch, nInputPlane, outputHeight, outputWidth]
    out = _35
  return (out, out)

)=====")
+ std::string(R"=====(def t(self: List[int]) -> List[int]:
  if torch.le(torch.len(self), 2):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  self_len = torch.len(self)
  if torch.eq(self_len, 0):
    _0 = annotate(List[int], [])
  else:
    if torch.eq(self_len, 1):
      _1 = [self[0]]
    else:
      _1 = [self[1], self[0]]
    _0 = _1
  return _0

def transpose(self: List[int],
    dim0: int,
    dim1: int) -> List[int]:
  ndims = torch.len(self)
  if torch.le(ndims, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = ndims
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(dim0, min):
    _0 = True
  else:
    _0 = torch.gt(dim0, max)
  if torch.__not__(_0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim0, 0):
    dim00 = torch.add(dim0, dim_post_expr)
  else:
    dim00 = dim0
  if torch.le(ndims, 0):
    dim_post_expr0 = 1
  else:
    dim_post_expr0 = ndims
  min0 = torch.neg(dim_post_expr0)
  max0 = torch.sub(dim_post_expr0, 1)
  if torch.lt(dim1, min0):
    _1 = True
  else:
    _1 = torch.gt(dim1, max0)
  if torch.__not__(_1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(dim1, 0):
    dim10 = torch.add(dim1, dim_post_expr0)
  else:
    dim10 = dim1
  if torch.eq(dim00, dim10):
    out = annotate(List[int], [])
    for _3 in range(torch.len(self)):
      elem = self[_3]
      _4 = torch.append(out, elem)
    _2 = out
  else:
    out0 = annotate(List[int], [])
    for i in range(ndims):
      if torch.eq(i, dim00):
        _5 = torch.append(out0, self[dim10])
      else:
        if torch.eq(i, dim10):
          _6 = torch.append(out0, self[dim00])
        else:
          _7 = torch.append(out0, self[i])
    _2 = out0
  return _2

)=====")
+ std::string(R"=====(def conv1d(input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int) -> List[int]:
  if torch.eq(torch.len(weight), 3):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 3):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  k = torch.len(input)
  weight_dim = torch.len(weight)
  non_negative = False
  for _0 in range(torch.len(padding)):
    val = padding[_0]
    if torch.lt(val, 0):
      non_negative0 = True
    else:
      non_negative0 = non_negative
    non_negative = non_negative0
  if torch.__not__(non_negative):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  non_negative1 = False
  for _1 in range(torch.len(stride)):
    val0 = stride[_1]
    if torch.lt(val0, 0):
      non_negative2 = True
    else:
      non_negative2 = non_negative1
    non_negative1 = non_negative2
  if torch.__not__(non_negative1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(weight_dim, k):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(weight[0], groups):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _2 = torch.eq(torch.remainder(weight[0], groups), 0)
  if _2:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _3 = torch.eq(input[1], torch.mul(weight[1], groups))
  if _3:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.__is__(bias, None):
    _4 = True
  else:
    bias0 = unchecked_cast(List[int], bias)
    if torch.eq(torch.len(bias0), 1):
      _5 = torch.eq(bias0[0], weight[0])
    else:
      _5 = False
    _4 = _5
  if _4:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  for _6 in range(torch.__range_length(2, k, 1)):
    i = torch.__derive_index(_6, 2, 1)
    _7 = input[i]
    _8 = torch.mul(padding[torch.sub(i, 2)], 2)
    _9 = torch.add(_7, _8)
    _10 = torch.mul(dilation[torch.sub(i, 2)], torch.sub(weight[i], 1))
    if torch.ge(_9, torch.add(_10, 1)):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  has_dilation = torch.gt(torch.len(dilation), 0)
  dim = torch.len(input)
  output_size = annotate(List[int], [])
  _11 = torch.append(output_size, input[0])
  _12 = torch.append(output_size, weight[0])
  for _13 in range(torch.__range_length(2, dim, 1)):
    d = torch.__derive_index(_13, 2, 1)
    if has_dilation:
      dilation_ = dilation[torch.sub(d, 2)]
    else:
      dilation_ = 1
    _14 = torch.mul(dilation_, torch.sub(weight[d], 1))
    kernel = torch.add(_14, 1)
    _15 = input[d]
    _16 = torch.mul(padding[torch.sub(d, 2)], 2)
    _17 = torch.sub(torch.add(_15, _16), kernel)
    _18 = torch.floordiv(_17, stride[torch.sub(d, 2)])
    _19 = torch.append(output_size, torch.add(_18, 1))
  return output_size

)=====")
+ std::string(R"=====(def conv2d(input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int) -> List[int]:
  if torch.eq(torch.len(weight), 4):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 4):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  k = torch.len(input)
  weight_dim = torch.len(weight)
  non_negative = False
  for _0 in range(torch.len(padding)):
    val = padding[_0]
    if torch.lt(val, 0):
      non_negative0 = True
    else:
      non_negative0 = non_negative
    non_negative = non_negative0
  if torch.__not__(non_negative):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  non_negative1 = False
  for _1 in range(torch.len(stride)):
    val0 = stride[_1]
    if torch.lt(val0, 0):
      non_negative2 = True
    else:
      non_negative2 = non_negative1
    non_negative1 = non_negative2
  if torch.__not__(non_negative1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(weight_dim, k):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(weight[0], groups):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _2 = torch.eq(torch.remainder(weight[0], groups), 0)
  if _2:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _3 = torch.eq(input[1], torch.mul(weight[1], groups))
  if _3:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.__is__(bias, None):
    _4 = True
  else:
    bias0 = unchecked_cast(List[int], bias)
    if torch.eq(torch.len(bias0), 1):
      _5 = torch.eq(bias0[0], weight[0])
    else:
      _5 = False
    _4 = _5
  if _4:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  for _6 in range(torch.__range_length(2, k, 1)):
    i = torch.__derive_index(_6, 2, 1)
    _7 = input[i]
    _8 = torch.mul(padding[torch.sub(i, 2)], 2)
    _9 = torch.add(_7, _8)
    _10 = torch.mul(dilation[torch.sub(i, 2)], torch.sub(weight[i], 1))
    if torch.ge(_9, torch.add(_10, 1)):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  has_dilation = torch.gt(torch.len(dilation), 0)
  dim = torch.len(input)
  output_size = annotate(List[int], [])
  _11 = torch.append(output_size, input[0])
  _12 = torch.append(output_size, weight[0])
  for _13 in range(torch.__range_length(2, dim, 1)):
    d = torch.__derive_index(_13, 2, 1)
    if has_dilation:
      dilation_ = dilation[torch.sub(d, 2)]
    else:
      dilation_ = 1
    _14 = torch.mul(dilation_, torch.sub(weight[d], 1))
    kernel = torch.add(_14, 1)
    _15 = input[d]
    _16 = torch.mul(padding[torch.sub(d, 2)], 2)
    _17 = torch.sub(torch.add(_15, _16), kernel)
    _18 = torch.floordiv(_17, stride[torch.sub(d, 2)])
    _19 = torch.append(output_size, torch.add(_18, 1))
  return output_size

)=====")
+ std::string(R"=====(def batch_norm(input: List[int],
    weight: Optional[List[int]],
    bias: Optional[List[int]],
    running_mean: Optional[List[int]],
    running_var: Optional[List[int]],
    training: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool) -> List[int]:
  out = annotate(List[int], [])
  for _0 in range(torch.len(input)):
    elem = input[_0]
    _1 = torch.append(out, elem)
  return out

)=====")
+ std::string(R"=====(def conv3d(input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int) -> List[int]:
  if torch.eq(torch.len(weight), 5):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 5):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  k = torch.len(input)
  weight_dim = torch.len(weight)
  non_negative = False
  for _0 in range(torch.len(padding)):
    val = padding[_0]
    if torch.lt(val, 0):
      non_negative0 = True
    else:
      non_negative0 = non_negative
    non_negative = non_negative0
  if torch.__not__(non_negative):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  non_negative1 = False
  for _1 in range(torch.len(stride)):
    val0 = stride[_1]
    if torch.lt(val0, 0):
      non_negative2 = True
    else:
      non_negative2 = non_negative1
    non_negative1 = non_negative2
  if torch.__not__(non_negative1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(weight_dim, k):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.ge(weight[0], groups):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _2 = torch.eq(torch.remainder(weight[0], groups), 0)
  if _2:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _3 = torch.eq(input[1], torch.mul(weight[1], groups))
  if _3:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.__is__(bias, None):
    _4 = True
  else:
    bias0 = unchecked_cast(List[int], bias)
    if torch.eq(torch.len(bias0), 1):
      _5 = torch.eq(bias0[0], weight[0])
    else:
      _5 = False
    _4 = _5
  if _4:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  for _6 in range(torch.__range_length(2, k, 1)):
    i = torch.__derive_index(_6, 2, 1)
    _7 = input[i]
    _8 = torch.mul(padding[torch.sub(i, 2)], 2)
    _9 = torch.add(_7, _8)
    _10 = torch.mul(dilation[torch.sub(i, 2)], torch.sub(weight[i], 1))
    if torch.ge(_9, torch.add(_10, 1)):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  has_dilation = torch.gt(torch.len(dilation), 0)
  dim = torch.len(input)
  output_size = annotate(List[int], [])
  _11 = torch.append(output_size, input[0])
  _12 = torch.append(output_size, weight[0])
  for _13 in range(torch.__range_length(2, dim, 1)):
    d = torch.__derive_index(_13, 2, 1)
    if has_dilation:
      dilation_ = dilation[torch.sub(d, 2)]
    else:
      dilation_ = 1
    _14 = torch.mul(dilation_, torch.sub(weight[d], 1))
    kernel = torch.add(_14, 1)
    _15 = input[d]
    _16 = torch.mul(padding[torch.sub(d, 2)], 2)
    _17 = torch.sub(torch.add(_15, _16), kernel)
    _18 = torch.floordiv(_17, stride[torch.sub(d, 2)])
    _19 = torch.append(output_size, torch.add(_18, 1))
  return output_size

)=====")
+ std::string(R"=====(def conv_backwards(grad_output: List[int],
    input: List[int],
    weight: List[int],
    biases: Optional[List[int]]) -> Tuple[List[int], List[int], List[int]]:
  out = annotate(List[int], [])
  for _0 in range(torch.len(input)):
    elem = input[_0]
    _1 = torch.append(out, elem)
  out0 = annotate(List[int], [])
  for _2 in range(torch.len(weight)):
    elem0 = weight[_2]
    _3 = torch.append(out0, elem0)
  return (out, out0, [grad_output[1]])

)=====")
+ std::string(R"=====(def conv_forwards(input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int) -> List[int]:
  has_dilation = torch.gt(torch.len(dilation), 0)
  has_output_padding = torch.gt(torch.len(output_padding), 0)
  dim = torch.len(input)
  output_size = annotate(List[int], [])
  if transposed:
    weight_output_channels_dim = 1
  else:
    weight_output_channels_dim = 0
  _0 = torch.append(output_size, input[0])
  if transposed:
    _1 = torch.mul(weight[weight_output_channels_dim], groups)
    _2 = torch.append(output_size, _1)
  else:
    _3 = torch.append(output_size, weight[weight_output_channels_dim])
  for _4 in range(torch.__range_length(2, dim, 1)):
    d = torch.__derive_index(_4, 2, 1)
    if has_dilation:
      dilation_ = dilation[torch.sub(d, 2)]
    else:
      dilation_ = 1
    if has_output_padding:
      output_padding_ = output_padding[torch.sub(d, 2)]
    else:
      output_padding_ = 0
    if transposed:
      kernel = torch.mul(dilation_, torch.sub(weight[d], 1))
      _5 = torch.mul(torch.sub(input[d], 1), stride[torch.sub(d, 2)])
      _6 = torch.mul(padding[torch.sub(d, 2)], 2)
      _7 = torch.add(torch.sub(_5, _6), kernel)
      _8 = torch.add(torch.add(_7, output_padding_), 1)
      _9 = torch.append(output_size, _8)
    else:
      _10 = torch.mul(dilation_, torch.sub(weight[d], 1))
      kernel0 = torch.add(_10, 1)
      _11 = input[d]
      _12 = torch.mul(padding[torch.sub(d, 2)], 2)
      _13 = torch.sub(torch.add(_11, _12), kernel0)
      _14 = torch.floordiv(_13, stride[torch.sub(d, 2)])
      _15 = torch.append(output_size, torch.add(_14, 1))
  return output_size

)=====")
+ std::string(R"=====(def _conv_forwards(input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int,
    benchmark: bool,
    deterministic: bool,
    cudnn_enabled: bool,
    allow_tf32: bool) -> List[int]:
  has_dilation = torch.gt(torch.len(dilation), 0)
  has_output_padding = torch.gt(torch.len(output_padding), 0)
  dim = torch.len(input)
  output_size = annotate(List[int], [])
  if transposed:
    weight_output_channels_dim = 1
  else:
    weight_output_channels_dim = 0
  _0 = torch.append(output_size, input[0])
  if transposed:
    _1 = torch.mul(weight[weight_output_channels_dim], groups)
    _2 = torch.append(output_size, _1)
  else:
    _3 = torch.append(output_size, weight[weight_output_channels_dim])
  for _4 in range(torch.__range_length(2, dim, 1)):
    d = torch.__derive_index(_4, 2, 1)
    if has_dilation:
      dilation_ = dilation[torch.sub(d, 2)]
    else:
      dilation_ = 1
    if has_output_padding:
      output_padding_ = output_padding[torch.sub(d, 2)]
    else:
      output_padding_ = 0
    if transposed:
      kernel = torch.mul(dilation_, torch.sub(weight[d], 1))
      _5 = torch.mul(torch.sub(input[d], 1), stride[torch.sub(d, 2)])
      _6 = torch.mul(padding[torch.sub(d, 2)], 2)
      _7 = torch.add(torch.sub(_5, _6), kernel)
      _8 = torch.add(torch.add(_7, output_padding_), 1)
      _9 = torch.append(output_size, _8)
    else:
      _10 = torch.mul(dilation_, torch.sub(weight[d], 1))
      kernel0 = torch.add(_10, 1)
      _11 = input[d]
      _12 = torch.mul(padding[torch.sub(d, 2)], 2)
      _13 = torch.sub(torch.add(_11, _12), kernel0)
      _14 = torch.floordiv(_13, stride[torch.sub(d, 2)])
      _15 = torch.append(output_size, torch.add(_14, 1))
  return output_size

)=====")
+ std::string(R"=====(def conv_transpose2d_input(input: List[int],
    weight: List[int],
    bias: Optional[List[int]]=None,
    stride: Optional[List[int]]=None,
    padding: Optional[List[int]]=None,
    output_padding: Optional[List[int]]=None,
    groups: int=1,
    dilation: Optional[List[int]]=None) -> List[int]:
  if torch.__is__(stride, None):
    stride0 = [1, 1]
  else:
    stride0 = unchecked_cast(List[int], stride)
  if torch.__is__(padding, None):
    padding0 = [0, 0]
  else:
    padding0 = unchecked_cast(List[int], padding)
  if torch.__is__(output_padding, None):
    output_padding0 = [0, 0]
  else:
    output_padding1 = unchecked_cast(List[int], output_padding)
    output_padding0 = output_padding1
  if torch.__is__(dilation, None):
    dilation0 = [1, 1]
  else:
    dilation0 = unchecked_cast(List[int], dilation)
  has_dilation = torch.gt(torch.len(dilation0), 0)
  dim = torch.len(input)
  output_size = annotate(List[int], [])
  _0 = torch.append(output_size, input[0])
  _1 = torch.append(output_size, torch.mul(weight[1], groups))
  for _2 in range(torch.__range_length(2, dim, 1)):
    d = torch.__derive_index(_2, 2, 1)
    if has_dilation:
      dilation_ = dilation0[torch.sub(d, 2)]
    else:
      dilation_ = 1
    kernel = torch.mul(dilation_, torch.sub(weight[d], 1))
    _3 = torch.mul(torch.sub(input[d], 1), stride0[torch.sub(d, 2)])
    _4 = torch.mul(padding0[torch.sub(d, 2)], 2)
    _5 = torch.add(torch.sub(_3, _4), kernel)
    _6 = torch.add(_5, output_padding0[torch.sub(d, 2)])
    _7 = torch.append(output_size, torch.add(_6, 1))
  return output_size

)=====")
+ std::string(R"=====(def flatten(input: List[int],
    start_dim: int,
    end_dim: int) -> List[int]:
  _0 = torch.len(input)
  if torch.le(_0, 0):
    dim_post_expr = 1
  else:
    dim_post_expr = _0
  min = torch.neg(dim_post_expr)
  max = torch.sub(dim_post_expr, 1)
  if torch.lt(start_dim, min):
    _1 = True
  else:
    _1 = torch.gt(start_dim, max)
  if torch.__not__(_1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(start_dim, 0):
    start_dim0 = torch.add(start_dim, dim_post_expr)
  else:
    start_dim0 = start_dim
  _2 = torch.len(input)
  if torch.le(_2, 0):
    dim_post_expr0 = 1
  else:
    dim_post_expr0 = _2
  min0 = torch.neg(dim_post_expr0)
  max0 = torch.sub(dim_post_expr0, 1)
  if torch.lt(end_dim, min0):
    _3 = True
  else:
    _3 = torch.gt(end_dim, max0)
  if torch.__not__(_3):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.lt(end_dim, 0):
    end_dim0 = torch.add(end_dim, dim_post_expr0)
  else:
    end_dim0 = end_dim
  if torch.le(start_dim0, end_dim0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(torch.len(input), 0):
    _4 = [1]
  else:
    if torch.eq(start_dim0, end_dim0):
      out = annotate(List[int], [])
      for _6 in range(torch.len(input)):
        elem = input[_6]
        _7 = torch.append(out, elem)
      _5 = out
    else:
      _8 = torch.__range_length(start_dim0, torch.add(end_dim0, 1), 1)
      slice_numel = 1
      for _9 in range(_8):
        i = torch.__derive_index(_9, start_dim0, 1)
        slice_numel0 = torch.mul(slice_numel, input[i])
        slice_numel = slice_numel0
      shape = annotate(List[int], [])
      for i0 in range(start_dim0):
        _10 = torch.append(shape, input[i0])
      _11 = torch.append(shape, slice_numel)
      _12 = torch.add(end_dim0, 1)
      _13 = torch.__range_length(_12, torch.len(input), 1)
      for _14 in range(_13):
        i1 = torch.__derive_index(_14, _12, 1)
        _15 = torch.append(shape, input[i1])
      _5 = shape
    _4 = _5
  return _4

)=====")
+ std::string(R"=====(def cat(tensors: List[List[int]],
    dim: int) -> List[int]:
  _0 = "AssertionError: Tensors must have same number of dimensions"
  _1 = "AssertionError: Sizes of tensors must match except in dimension"
  for _2 in range(torch.len(tensors)):
    tensor = tensors[_2]
    if torch.gt(torch.len(tensor), 0):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  out_dim: Optional[int] = None
  for _3 in range(torch.len(tensors)):
    size = tensors[_3]
    if torch.eq(torch.len(size), 1):
      _4 = torch.eq(size[0], 0)
    else:
      _4 = False
    if torch.__not__(_4):
      if torch.__is__(out_dim, None):
        _5 = torch.len(size)
        if torch.le(_5, 0):
          dim_post_expr = 1
        else:
          dim_post_expr = _5
        min = torch.neg(dim_post_expr)
        max = torch.sub(dim_post_expr, 1)
        if torch.lt(dim, min):
          _6 = True
        else:
          _6 = torch.gt(dim, max)
        if torch.__not__(_6):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        if torch.lt(dim, 0):
          out_dim2 = torch.add(dim, dim_post_expr)
        else:
          out_dim2 = dim
        out_dim1 = out_dim2
      else:
        out_dim1 = unchecked_cast(int, out_dim)
      out_dim0 : Optional[int] = out_dim1
    else:
      out_dim0 = out_dim
    out_dim = out_dim0
  if torch.__is__(out_dim, None):
    dim0 = dim
  else:
    dim0 = unchecked_cast(int, out_dim)
  if torch.gt(torch.len(tensors), 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  not_skipped_tensor: Optional[List[int]] = None
  for _7 in range(torch.len(tensors)):
    tensor0 = tensors[_7]
    numel = 1
    for _8 in range(torch.len(tensor0)):
      elem = tensor0[_8]
      numel = torch.mul(numel, elem)
    if torch.eq(numel, 0):
      _9 = torch.eq(torch.len(tensor0), 1)
    else:
      _9 = False
    if torch.__not__(_9):
      not_skipped_tensor0 : Optional[List[int]] = tensor0
    else:
      not_skipped_tensor0 = not_skipped_tensor
    not_skipped_tensor = not_skipped_tensor0
  _10 = torch.__is__(not_skipped_tensor, None)
  if _10:
    _11 = [0]
  else:
    not_skipped_tensor1 = unchecked_cast(List[int], not_skipped_tensor)
    cat_dim_size = 0
    for i in range(torch.len(tensors)):
      tensor1 = tensors[i]
      numel0 = 1
      for _12 in range(torch.len(tensor1)):
        elem0 = tensor1[_12]
        numel0 = torch.mul(numel0, elem0)
      if torch.eq(numel0, 0):
        _13 = torch.eq(torch.len(tensor1), 1)
      else:
        _13 = False
      if torch.__not__(_13):
        first_dims = torch.len(not_skipped_tensor1)
        second_dims = torch.len(tensor1)
        _14 = torch.eq(first_dims, second_dims)
        if _14:
          pass
        else:
          ops.prim.RaiseException(_0)
        _15 = torch.__range_length(0, first_dims, 1)
        for _16 in range(_15):
          dim1 = torch.__derive_index(_16, 0, 1)
          if torch.ne(dim1, dim0):
            _17 = torch.eq(not_skipped_tensor1[dim1], tensor1[dim1])
            if _17:
              pass
            else:
              ops.prim.RaiseException(_1)
          else:
            pass
        cat_dim_size1 = torch.add(cat_dim_size, tensor1[dim0])
        cat_dim_size0 = cat_dim_size1
      else:
        cat_dim_size0 = cat_dim_size
      cat_dim_size = cat_dim_size0
    result_size = annotate(List[int], [])
    for _18 in range(torch.len(not_skipped_tensor1)):
      elem1 = not_skipped_tensor1[_18]
      _19 = torch.append(result_size, elem1)
    _20 = torch._set_item(result_size, dim0, cat_dim_size)
    _11 = result_size
  return _11

)=====")
+ std::string(R"=====(def stack(tensors: List[List[int]],
    dim: int) -> List[int]:
  _0 = "AssertionError: Tensors must have same number of dimensions"
  _1 = "AssertionError: Sizes of tensors must match except in dimension"
  unsqueezed_tensors = annotate(List[List[int]], [])
  for _2 in range(torch.len(tensors)):
    tensor = tensors[_2]
    _3 = torch.add(torch.len(tensor), 1)
    if torch.le(_3, 0):
      dim_post_expr = 1
    else:
      dim_post_expr = _3
    min = torch.neg(dim_post_expr)
    max = torch.sub(dim_post_expr, 1)
    if torch.lt(dim, min):
      _4 = True
    else:
      _4 = torch.gt(dim, max)
    if torch.__not__(_4):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    if torch.lt(dim, 0):
      dim0 = torch.add(dim, dim_post_expr)
    else:
      dim0 = dim
    unsqueezed = annotate(List[int], [])
    for _5 in range(torch.len(tensor)):
      elem = tensor[_5]
      _6 = torch.append(unsqueezed, elem)
    torch.insert(unsqueezed, dim0, 1)
    _7 = torch.append(unsqueezed_tensors, unsqueezed)
  for _8 in range(torch.len(unsqueezed_tensors)):
    tensor0 = unsqueezed_tensors[_8]
    if torch.gt(torch.len(tensor0), 0):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
  out_dim: Optional[int] = None
  for _9 in range(torch.len(unsqueezed_tensors)):
    size = unsqueezed_tensors[_9]
    if torch.eq(torch.len(size), 1):
      _10 = torch.eq(size[0], 0)
    else:
      _10 = False
    if torch.__not__(_10):
      if torch.__is__(out_dim, None):
        _11 = torch.len(size)
        if torch.le(_11, 0):
          dim_post_expr0 = 1
        else:
          dim_post_expr0 = _11
        min0 = torch.neg(dim_post_expr0)
        max0 = torch.sub(dim_post_expr0, 1)
        if torch.lt(dim, min0):
          _12 = True
        else:
          _12 = torch.gt(dim, max0)
        if torch.__not__(_12):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        if torch.lt(dim, 0):
          dim1 = torch.add(dim, dim_post_expr0)
          out_dim2 = dim1
        else:
          out_dim2 = dim
        out_dim1 = out_dim2
      else:
        out_dim1 = unchecked_cast(int, out_dim)
      out_dim0 : Optional[int] = out_dim1
    else:
      out_dim0 = out_dim
    out_dim = out_dim0
  if torch.__is__(out_dim, None):
    dim2 = dim
  else:
    dim2 = unchecked_cast(int, out_dim)
  _13 = torch.gt(torch.len(unsqueezed_tensors), 0)
  if _13:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  not_skipped_tensor: Optional[List[int]] = None
  for _14 in range(torch.len(unsqueezed_tensors)):
    tensor1 = unsqueezed_tensors[_14]
    numel = 1
    for _15 in range(torch.len(tensor1)):
      elem0 = tensor1[_15]
      numel = torch.mul(numel, elem0)
    if torch.eq(numel, 0):
      _16 = torch.eq(torch.len(tensor1), 1)
    else:
      _16 = False
    if torch.__not__(_16):
      not_skipped_tensor0 : Optional[List[int]] = tensor1
    else:
      not_skipped_tensor0 = not_skipped_tensor
    not_skipped_tensor = not_skipped_tensor0
  _17 = torch.__is__(not_skipped_tensor, None)
  if _17:
    _18 = [0]
  else:
    not_skipped_tensor1 = unchecked_cast(List[int], not_skipped_tensor)
    cat_dim_size = 0
    for i in range(torch.len(unsqueezed_tensors)):
      tensor2 = unsqueezed_tensors[i]
      numel0 = 1
      for _19 in range(torch.len(tensor2)):
        elem1 = tensor2[_19]
        numel0 = torch.mul(numel0, elem1)
      if torch.eq(numel0, 0):
        _20 = torch.eq(torch.len(tensor2), 1)
      else:
        _20 = False
      if torch.__not__(_20):
        first_dims = torch.len(not_skipped_tensor1)
        second_dims = torch.len(tensor2)
        _21 = torch.eq(first_dims, second_dims)
        if _21:
          pass
        else:
          ops.prim.RaiseException(_0)
        _22 = torch.__range_length(0, first_dims, 1)
        for _23 in range(_22):
          dim3 = torch.__derive_index(_23, 0, 1)
          if torch.ne(dim3, dim2):
            _24 = torch.eq(not_skipped_tensor1[dim3], tensor2[dim3])
            if _24:
              pass
            else:
              ops.prim.RaiseException(_1)
          else:
            pass
        cat_dim_size1 = torch.add(cat_dim_size, tensor2[dim2])
        cat_dim_size0 = cat_dim_size1
      else:
        cat_dim_size0 = cat_dim_size
      cat_dim_size = cat_dim_size0
    result_size = annotate(List[int], [])
    for _25 in range(torch.len(not_skipped_tensor1)):
      elem2 = not_skipped_tensor1[_25]
      _26 = torch.append(result_size, elem2)
    _27 = torch._set_item(result_size, dim2, cat_dim_size)
    _18 = result_size
  return _18

)=====")
+ std::string(R"=====(def permute(input: List[int],
    dims: List[int]) -> List[int]:
  _0 = torch.eq(torch.len(input), torch.len(dims))
  if _0:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  ndim = torch.len(dims)
  seen_dims = annotate(List[int], [])
  newSizes = annotate(List[int], [])
  for i in range(ndim):
    _1 = dims[i]
    if torch.le(ndim, 0):
      dim_post_expr = 1
    else:
      dim_post_expr = ndim
    min = torch.neg(dim_post_expr)
    max = torch.sub(dim_post_expr, 1)
    if torch.lt(_1, min):
      _2 = True
    else:
      _2 = torch.gt(_1, max)
    if torch.__not__(_2):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    if torch.lt(_1, 0):
      dim = torch.add(_1, dim_post_expr)
    else:
      dim = _1
    _3 = torch.append(seen_dims, dim)
    _4 = torch.append(newSizes, input[dim])
  for _5 in range(torch.__range_length(1, ndim, 1)):
    i0 = torch.__derive_index(_5, 1, 1)
    for j in range(i0):
      _6 = torch.ne(seen_dims[i0], seen_dims[j])
      if _6:
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
  return newSizes

)=====")
+ std::string(R"=====(def movedim(self: List[int],
    source: List[int],
    destination: List[int]) -> List[int]:
  self_dim = torch.len(self)
  if torch.le(self_dim, 1):
    _0 = self
  else:
    normalized_src = annotate(List[int], [])
    normalized_dst = annotate(List[int], [])
    for i in range(torch.len(source)):
      _1 = source[i]
      if torch.le(self_dim, 0):
        dim_post_expr = 1
      else:
        dim_post_expr = self_dim
      min = torch.neg(dim_post_expr)
      max = torch.sub(dim_post_expr, 1)
      if torch.lt(_1, min):
        _2 = True
      else:
        _2 = torch.gt(_1, max)
      if torch.__not__(_2):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      if torch.lt(_1, 0):
        dim = torch.add(_1, dim_post_expr)
      else:
        dim = _1
      _3 = torch.append(normalized_src, dim)
      _4 = destination[i]
      if torch.le(self_dim, 0):
        dim_post_expr0 = 1
      else:
        dim_post_expr0 = self_dim
      min0 = torch.neg(dim_post_expr0)
      max0 = torch.sub(dim_post_expr0, 1)
      if torch.lt(_4, min0):
        _5 = True
      else:
        _5 = torch.gt(_4, max0)
      if torch.__not__(_5):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      if torch.lt(_4, 0):
        dim0 = torch.add(_4, dim_post_expr0)
      else:
        dim0 = _4
      _6 = torch.append(normalized_dst, dim0)
    order = annotate(List[int], [])
    for i0 in range(self_dim):
      _7 = torch.append(order, -1)
    src_dims = annotate(List[int], [])
    for i1 in range(self_dim):
      _8 = torch.append(src_dims, i1)
    dst_dims = annotate(List[int], [])
    for i2 in range(self_dim):
      _9 = torch.append(dst_dims, i2)
    for i3 in range(torch.len(source)):
      _10 = normalized_src[i3]
      _11 = torch._set_item(order, normalized_dst[i3], _10)
      _12 = torch._set_item(src_dims, normalized_src[i3], -1)
      _13 = torch._set_item(dst_dims, normalized_dst[i3], -1)
    source_dims = annotate(List[int], [])
    destination_dims = annotate(List[int], [])
    for _14 in range(torch.len(src_dims)):
      ele = src_dims[_14]
      if torch.ne(ele, -1):
        _15 = torch.append(source_dims, ele)
      else:
        pass
    for _16 in range(torch.len(dst_dims)):
      ele0 = dst_dims[_16]
      if torch.ne(ele0, -1):
        _17 = torch.append(destination_dims, ele0)
      else:
        pass
    rest_dim = torch.sub(self_dim, torch.len(source))
    for i4 in range(rest_dim):
      _18 = source_dims[i4]
      _19 = torch._set_item(order, destination_dims[i4], _18)
    _20 = torch.eq(torch.len(self), torch.len(order))
    if _20:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    ndim = torch.len(order)
    seen_dims = annotate(List[int], [])
    newSizes = annotate(List[int], [])
    for i5 in range(ndim):
      _21 = order[i5]
      if torch.le(ndim, 0):
        dim_post_expr1 = 1
      else:
        dim_post_expr1 = ndim
      min1 = torch.neg(dim_post_expr1)
      max1 = torch.sub(dim_post_expr1, 1)
      if torch.lt(_21, min1):
        _22 = True
      else:
        _22 = torch.gt(_21, max1)
      if torch.__not__(_22):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      if torch.lt(_21, 0):
        dim1 = torch.add(_21, dim_post_expr1)
      else:
        dim1 = _21
      _23 = torch.append(seen_dims, dim1)
      _24 = torch.append(newSizes, self[dim1])
    for _25 in range(torch.__range_length(1, ndim, 1)):
      i6 = torch.__derive_index(_25, 1, 1)
      for j in range(i6):
        _26 = torch.ne(seen_dims[i6], seen_dims[j])
        if _26:
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
    _0 = newSizes
  return _0

)=====")
+ std::string(R"=====(def view(self: List[int],
    sizes: List[int]) -> List[int]:
  _0 = "AssertionError: only one dimension can be inferred"
  _1 = "AssertionError: invalid shape dimensions"
  numel = 1
  for _2 in range(torch.len(self)):
    elem = self[_2]
    numel = torch.mul(numel, elem)
  _3 = uninitialized(int)
  newsize = 1
  infer_dim: Optional[int] = None
  for dim in range(torch.len(sizes)):
    if torch.eq(sizes[dim], -1):
      if torch.__isnot__(infer_dim, None):
        ops.prim.RaiseException(_0)
      else:
        pass
      newsize0, infer_dim0 = newsize, dim
    else:
      if torch.ge(sizes[dim], 0):
        newsize1 = torch.mul(newsize, sizes[dim])
      else:
        ops.prim.RaiseException(_1)
        newsize1 = _3
      newsize0, infer_dim0 = newsize1, infer_dim
    newsize, infer_dim = newsize0, infer_dim0
  if torch.eq(numel, newsize):
    _4, infer_dim1 = True, infer_dim
  else:
    if torch.__isnot__(infer_dim, None):
      infer_dim3 = unchecked_cast(int, infer_dim)
      _5, infer_dim2 = torch.gt(newsize, 0), infer_dim3
    else:
      _5, infer_dim2 = False, infer_dim
    if _5:
      infer_dim5 = unchecked_cast(int, infer_dim2)
      _7 = torch.eq(torch.remainder(numel, newsize), 0)
      _6, infer_dim4 = _7, infer_dim5
    else:
      _6, infer_dim4 = False, infer_dim2
    _4, infer_dim1 = _6, infer_dim4
  if torch.__not__(_4):
    ops.prim.RaiseException("AssertionError: invalid shape")
  else:
    pass
  out = annotate(List[int], [])
  for _8 in range(torch.len(sizes)):
    elem0 = sizes[_8]
    _9 = torch.append(out, elem0)
  if torch.__isnot__(infer_dim1, None):
    infer_dim6 = unchecked_cast(int, infer_dim1)
    _10 = torch._set_item(out, infer_dim6, torch.floordiv(numel, newsize))
  else:
    pass
  return out

)=====")
+ std::string(R"=====(def expand(self: List[int],
    sizes: List[int]) -> List[int]:
  _0 = torch.ge(torch.len(sizes), torch.len(self))
  if _0:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  ndim = torch.len(sizes)
  tensor_dim = torch.len(self)
  if torch.eq(ndim, 0):
    out = annotate(List[int], [])
    for _2 in range(torch.len(sizes)):
      elem = sizes[_2]
      _3 = torch.append(out, elem)
    _1 = out
  else:
    out0 = annotate(List[int], [])
    for i in range(ndim):
      offset = torch.sub(torch.sub(ndim, 1), i)
      dim = torch.sub(torch.sub(tensor_dim, 1), offset)
      if torch.ge(dim, 0):
        size = self[dim]
      else:
        size = 1
      targetSize = sizes[i]
      if torch.eq(targetSize, -1):
        if torch.ge(dim, 0):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        targetSize0 = size
      else:
        targetSize0 = targetSize
      if torch.ne(size, targetSize0):
        if torch.eq(size, 1):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        size0 = targetSize0
      else:
        size0 = size
      _4 = torch.append(out0, size0)
    _1 = out0
  return _1

)=====")
+ std::string(R"=====(def expand_one_unused(self: List[int],
    sizes: List[int],
    inp0: Any) -> List[int]:
  _0 = torch.ge(torch.len(sizes), torch.len(self))
  if _0:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  ndim = torch.len(sizes)
  tensor_dim = torch.len(self)
  if torch.eq(ndim, 0):
    out = annotate(List[int], [])
    for _2 in range(torch.len(sizes)):
      elem = sizes[_2]
      _3 = torch.append(out, elem)
    _1 = out
  else:
    out0 = annotate(List[int], [])
    for i in range(ndim):
      offset = torch.sub(torch.sub(ndim, 1), i)
      dim = torch.sub(torch.sub(tensor_dim, 1), offset)
      if torch.ge(dim, 0):
        size = self[dim]
      else:
        size = 1
      targetSize = sizes[i]
      if torch.eq(targetSize, -1):
        if torch.ge(dim, 0):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        targetSize0 = size
      else:
        targetSize0 = targetSize
      if torch.ne(size, targetSize0):
        if torch.eq(size, 1):
          pass
        else:
          ops.prim.RaiseException("AssertionError: ")
        size0 = targetSize0
      else:
        size0 = size
      _4 = torch.append(out0, size0)
    _1 = out0
  return _1

)=====")
+ std::string(R"=====(def sum_mean_dim(self: List[int],
    opt_dims: Optional[List[int]],
    keep_dim: bool,
    dt: Any) -> List[int]:
  out = annotate(List[int], [])
  if torch.__is__(opt_dims, None):
    _0, opt_dims0 = True, opt_dims
  else:
    opt_dims1 = unchecked_cast(List[int], opt_dims)
    _0, opt_dims0 = torch.eq(torch.len(opt_dims1), 0), opt_dims1
  if _0:
    _1 = torch.len(self)
    dims0 = annotate(List[int], [])
    for _2 in range(_1):
      _3 = torch.append(dims0, _2)
    dims = dims0
  else:
    opt_dims2 = unchecked_cast(List[int], opt_dims0)
    dims = opt_dims2
  for idx in range(torch.len(self)):
    is_mean_dim = False
    for _4 in range(torch.len(dims)):
      reduce_dim = dims[_4]
      _5 = torch.len(self)
      if torch.le(_5, 0):
        dim_post_expr = 1
      else:
        dim_post_expr = _5
      min = torch.neg(dim_post_expr)
      max = torch.sub(dim_post_expr, 1)
      if torch.lt(reduce_dim, min):
        _6 = True
      else:
        _6 = torch.gt(reduce_dim, max)
      if torch.__not__(_6):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      if torch.lt(reduce_dim, 0):
        dim0 = torch.add(reduce_dim, dim_post_expr)
        dim = dim0
      else:
        dim = reduce_dim
      if torch.eq(idx, dim):
        is_mean_dim0 = True
      else:
        is_mean_dim0 = is_mean_dim
      is_mean_dim = is_mean_dim0
    if is_mean_dim:
      if keep_dim:
        _7 = torch.append(out, 1)
      else:
        pass
    else:
      _8 = torch.append(out, self[idx])
  return out

)=====")
+ std::string(R"=====(def max_dim(self: List[int],
    dim: int,
    keep_dim: bool) -> Tuple[List[int], List[int]]:
  _0 = [dim]
  out = annotate(List[int], [])
  for idx in range(torch.len(self)):
    is_mean_dim = False
    for _1 in range(torch.len(_0)):
      reduce_dim = _0[_1]
      _2 = torch.len(self)
      if torch.le(_2, 0):
        dim_post_expr = 1
      else:
        dim_post_expr = _2
      min = torch.neg(dim_post_expr)
      max = torch.sub(dim_post_expr, 1)
      if torch.lt(reduce_dim, min):
        _3 = True
      else:
        _3 = torch.gt(reduce_dim, max)
      if torch.__not__(_3):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      if torch.lt(reduce_dim, 0):
        dim1 = torch.add(reduce_dim, dim_post_expr)
        dim0 = dim1
      else:
        dim0 = reduce_dim
      if torch.eq(idx, dim0):
        is_mean_dim0 = True
      else:
        is_mean_dim0 = is_mean_dim
      is_mean_dim = is_mean_dim0
    if is_mean_dim:
      if keep_dim:
        _4 = torch.append(out, 1)
      else:
        pass
    else:
      _5 = torch.append(out, self[idx])
  return (out, out)

)=====")
+ std::string(R"=====(def addmm(self: List[int],
    mat1: List[int],
    mat2: List[int],
    beta: Any,
    alpha: Any) -> List[int]:
  _0 = "AssertionError: self must be a matrix"
  _1 = "AssertionError: mat2 must be a matrix"
  _2 = "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
  if torch.eq(torch.len(mat1), 2):
    pass
  else:
    ops.prim.RaiseException(_0)
  if torch.eq(torch.len(mat2), 2):
    pass
  else:
    ops.prim.RaiseException(_1)
  if torch.eq(mat1[1], mat2[0]):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _3 = [mat1[0], mat2[1]]
  dimsA = torch.len(self)
  ndim = ops.prim.max(dimsA, 2)
  expandedSizes = annotate(List[int], [])
  for i in range(ndim):
    offset = torch.sub(torch.sub(ndim, 1), i)
    dimA = torch.sub(torch.sub(dimsA, 1), offset)
    dimB = torch.sub(1, offset)
    if torch.ge(dimA, 0):
      sizeA = self[dimA]
    else:
      sizeA = 1
    if torch.ge(dimB, 0):
      sizeB = _3[dimB]
    else:
      sizeB = 1
    if torch.ne(sizeA, sizeB):
      _4 = torch.ne(sizeA, 1)
    else:
      _4 = False
    if _4:
      _5 = torch.ne(sizeB, 1)
    else:
      _5 = False
    if _5:
      _6 = torch.add("AssertionError: ", torch.format(_2, sizeA, sizeB, i))
      ops.prim.RaiseException(_6)
    else:
      pass
    if torch.eq(sizeA, 1):
      _7 = sizeB
    else:
      _7 = sizeA
    _8 = torch.append(expandedSizes, _7)
  return expandedSizes

)=====")
+ std::string(R"=====(def upsample_nearest2d(input: List[int],
    output_size: Optional[List[int]],
    scale_factors: Optional[List[float]]) -> List[int]:
  _0 = "AssertionError: Either output_size or scale_factors must be presented"
  _1 = "AssertionError: Must specify exactly one of output_size and scale_factors"
  _2 = uninitialized(Optional[List[float]])
  out = annotate(List[int], [])
  _3 = torch.append(out, input[0])
  _4 = torch.append(out, input[1])
  if torch.__is__(scale_factors, None):
    _5, scale_factors0 = torch.__is__(output_size, None), scale_factors
  else:
    scale_factors1 = unchecked_cast(List[float], scale_factors)
    _5, scale_factors0 = False, scale_factors1
  if _5:
    ops.prim.RaiseException(_0)
  else:
    pass
  if torch.__isnot__(output_size, None):
    output_size1 = unchecked_cast(List[int], output_size)
    if torch.__is__(scale_factors0, None):
      scale_factors3 : Optional[List[float]] = scale_factors0
    else:
      ops.prim.RaiseException(_1)
      scale_factors3 = _2
    _6 = torch.eq(torch.len(output_size1), 2)
    if _6:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    _7 = torch.append(out, output_size1[0])
    _8 = torch.append(out, output_size1[1])
    scale_factors2, output_size0 = scale_factors3, output_size1
  else:
    scale_factors2, output_size0 = scale_factors0, output_size
  if torch.__isnot__(scale_factors2, None):
    scale_factors4 = unchecked_cast(List[float], scale_factors2)
    if torch.__is__(output_size0, None):
      pass
    else:
      ops.prim.RaiseException(_1)
    _9 = torch.eq(torch.len(scale_factors4), 2)
    if _9:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    _10 = torch.mul(input[2], scale_factors4[0])
    _11 = torch.append(out, int(_10))
    _12 = torch.mul(input[3], scale_factors4[1])
    _13 = torch.append(out, int(_12))
  else:
    pass
  return out

)=====")
+ std::string(R"=====(def broadcast(a: List[int],
    b: List[int]) -> List[int]:
  _0 = "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
  dimsA = torch.len(a)
  dimsB = torch.len(b)
  ndim = ops.prim.max(dimsA, dimsB)
  expandedSizes = annotate(List[int], [])
  for i in range(ndim):
    offset = torch.sub(torch.sub(ndim, 1), i)
    dimA = torch.sub(torch.sub(dimsA, 1), offset)
    dimB = torch.sub(torch.sub(dimsB, 1), offset)
    if torch.ge(dimA, 0):
      sizeA = a[dimA]
    else:
      sizeA = 1
    if torch.ge(dimB, 0):
      sizeB = b[dimB]
    else:
      sizeB = 1
    if torch.ne(sizeA, sizeB):
      _1 = torch.ne(sizeA, 1)
    else:
      _1 = False
    if _1:
      _2 = torch.ne(sizeB, 1)
    else:
      _2 = False
    if _2:
      _3 = torch.add("AssertionError: ", torch.format(_0, sizeA, sizeB, i))
      ops.prim.RaiseException(_3)
    else:
      pass
    if torch.eq(sizeA, 1):
      _4 = sizeB
    else:
      _4 = sizeA
    _5 = torch.append(expandedSizes, _4)
  return expandedSizes

)=====")
+ std::string(R"=====(def argmax(self: List[int],
    dim: Optional[int]=None,
    keepdim: bool=False) -> List[int]:
  if torch.__is__(dim, None):
    _0 = annotate(List[int], [])
  else:
    dim0 = unchecked_cast(int, dim)
    _1 = torch.len(self)
    if torch.le(_1, 0):
      dim_post_expr = 1
    else:
      dim_post_expr = _1
    min = torch.neg(dim_post_expr)
    max = torch.sub(dim_post_expr, 1)
    if torch.lt(dim0, min):
      _2 = True
    else:
      _2 = torch.gt(dim0, max)
    if torch.__not__(_2):
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    if torch.lt(dim0, 0):
      dim1 = torch.add(dim0, dim_post_expr)
    else:
      dim1 = dim0
    out = annotate(List[int], [])
    _3 = [9223372036854775807, torch.len(self)]
    for i in range(ops.prim.min(_3)):
      self_dim = self[i]
      if torch.eq(i, dim1):
        if keepdim:
          _4 = torch.append(out, 1)
        else:
          pass
      else:
        _5 = torch.append(out, self_dim)
    _0 = out
  return _0

def bmm(self: List[int],
    mat2: List[int]) -> List[int]:
  _0 = "AssertionError: bmm only supports 3D tensors"
  _1 = "AssertionError: mismatching batch dimension"
  _2 = "AssertionError: mismatching contracting dimension"
  if torch.eq(torch.len(self), 3):
    pass
  else:
    ops.prim.RaiseException(_0)
  if torch.eq(torch.len(mat2), 3):
    pass
  else:
    ops.prim.RaiseException(_0)
  if torch.eq(self[0], mat2[0]):
    pass
  else:
    ops.prim.RaiseException(_1)
  if torch.eq(self[2], mat2[1]):
    pass
  else:
    ops.prim.RaiseException(_2)
  return [self[0], self[1], mat2[2]]

def _shape_as_tensor(self: List[int]) -> List[int]:
  return [torch.len(self)]

)=====")
+ std::string(R"=====(def topk(self: List[int],
    k: int,
    dim: int=-1) -> Tuple[List[int], List[int]]:
  _0 = "k ({}) is too big for dimension {} of size {}"
  if torch.eq(torch.len(self), 0):
    result = annotate(List[int], [])
  else:
    if torch.le(k, self[dim]):
      pass
    else:
      _1 = torch.format(_0, k, dim, self[dim])
      ops.prim.RaiseException(torch.add("AssertionError: ", _1))
    result0 = annotate(List[int], [])
    for _2 in range(torch.len(self)):
      elem = self[_2]
      _3 = torch.append(result0, elem)
    _4 = torch._set_item(result0, dim, k)
    result = result0
  return (result, result)

def nll_loss_forward(self: List[int],
    target: List[int],
    weight: Optional[List[int]],
    reduction: int) -> Tuple[List[int], List[int]]:
  self_dim = torch.len(self)
  target_dim = torch.len(target)
  if torch.lt(0, self_dim):
    _0 = torch.le(self_dim, 2)
  else:
    _0 = False
  if _0:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.le(target_dim, 1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(self_dim, 1):
    no_batch_dim = torch.eq(target_dim, 0)
  else:
    no_batch_dim = False
  if no_batch_dim:
    _1 = True
  else:
    _1 = torch.eq(self[0], target[0])
  if _1:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  n_classes = self[-1]
  if torch.__is__(weight, None):
    _2 = True
  else:
    weight0 = unchecked_cast(List[int], weight)
    if torch.eq(torch.len(weight0), 1):
      _3 = torch.eq(weight0[0], n_classes)
    else:
      _3 = False
    _2 = _3
  if _2:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(reduction, 0):
    _4 = torch.eq(self_dim, 2)
  else:
    _4 = False
  if _4:
    reduction_shape = [self[0]]
  else:
    reduction_shape = annotate(List[int], [])
  _5 = (reduction_shape, annotate(List[int], []))
  return _5

)=====")
+ std::string(R"=====(def native_layer_norm(input: List[int],
    normalized_shape: List[int]) -> Tuple[List[int], List[int], List[int]]:
  reduction_shape = annotate(List[int], [])
  num_unreduced_dimensions = torch.sub(torch.len(input), torch.len(normalized_shape))
  if torch.ge(num_unreduced_dimensions, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  for i in range(num_unreduced_dimensions):
    _0 = torch.append(reduction_shape, input[i])
  _1 = torch.__range_length(num_unreduced_dimensions, torch.len(input), 1)
  for _2 in range(_1):
    _3 = torch.append(reduction_shape, 1)
  out = annotate(List[int], [])
  for _4 in range(torch.len(input)):
    elem = input[_4]
    _5 = torch.append(out, elem)
  _6 = (out, reduction_shape, reduction_shape)
  return _6

def native_batch_norm(input: List[int],
    weight: Optional[List[int]],
    bias: Optional[List[int]],
    running_mean: Optional[List[int]],
    running_var: Optional[List[int]],
    training: bool) -> Tuple[List[int], List[int], List[int]]:
  if training:
    _size = [input[1]]
  else:
    _size = [0]
  out = annotate(List[int], [])
  for _0 in range(torch.len(input)):
    elem = input[_0]
    _1 = torch.append(out, elem)
  return (out, _size, _size)

def _batch_norm_with_update(input: List[int],
    weight: Optional[List[int]],
    bias: Optional[List[int]],
    running_mean: Optional[List[int]],
    running_var: Optional[List[int]]) -> Tuple[List[int], List[int], List[int], List[int]]:
  _size = [input[1]]
  out = annotate(List[int], [])
  for _0 in range(torch.len(input)):
    elem = input[_0]
    _1 = torch.append(out, elem)
  return (out, _size, _size, [0])

)=====")
+ std::string(R"=====(def cross_entropy_loss(self: List[int],
    target: List[int],
    weight: Optional[List[int]]=None,
    reduction: int=1,
    ignore_index: int=-100,
    label_smoothing: float=0.) -> List[int]:
  self_dim = torch.len(self)
  target_dim = torch.len(target)
  if torch.lt(0, self_dim):
    _0 = torch.le(self_dim, 2)
  else:
    _0 = False
  if _0:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.le(target_dim, 1):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(self_dim, 1):
    no_batch_dim = torch.eq(target_dim, 0)
  else:
    no_batch_dim = False
  if no_batch_dim:
    _1 = True
  else:
    _1 = torch.eq(self[0], target[0])
  if _1:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  n_classes = self[-1]
  if torch.__is__(weight, None):
    _2 = True
  else:
    weight0 = unchecked_cast(List[int], weight)
    if torch.eq(torch.len(weight0), 1):
      _3 = torch.eq(weight0[0], n_classes)
    else:
      _3 = False
    _2 = _3
  if _2:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  if torch.eq(reduction, 0):
    _4 = torch.eq(self_dim, 2)
  else:
    _4 = False
  if _4:
    reduction_shape = [self[0]]
  else:
    reduction_shape = annotate(List[int], [])
  _5 = (reduction_shape, annotate(List[int], []))
  return (_5)[0]

)=====")
+ std::string(R"=====(def broadcast_three(a: List[int],
    b: List[int],
    c: List[int]) -> List[int]:
  _0 = "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
  _1 = "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
  dimsA = torch.len(a)
  dimsB = torch.len(b)
  ndim = ops.prim.max(dimsA, dimsB)
  expandedSizes = annotate(List[int], [])
  for i in range(ndim):
    offset = torch.sub(torch.sub(ndim, 1), i)
    dimA = torch.sub(torch.sub(dimsA, 1), offset)
    dimB = torch.sub(torch.sub(dimsB, 1), offset)
    if torch.ge(dimA, 0):
      sizeA = a[dimA]
    else:
      sizeA = 1
    if torch.ge(dimB, 0):
      sizeB = b[dimB]
    else:
      sizeB = 1
    if torch.ne(sizeA, sizeB):
      _2 = torch.ne(sizeA, 1)
    else:
      _2 = False
    if _2:
      _3 = torch.ne(sizeB, 1)
    else:
      _3 = False
    if _3:
      _4 = torch.add("AssertionError: ", torch.format(_0, sizeA, sizeB, i))
      ops.prim.RaiseException(_4)
    else:
      pass
    if torch.eq(sizeA, 1):
      _5 = sizeB
    else:
      _5 = sizeA
    _6 = torch.append(expandedSizes, _5)
  dimsA0 = torch.len(expandedSizes)
  dimsB0 = torch.len(c)
  ndim0 = ops.prim.max(dimsA0, dimsB0)
  expandedSizes0 = annotate(List[int], [])
  for i0 in range(ndim0):
    offset0 = torch.sub(torch.sub(ndim0, 1), i0)
    dimA0 = torch.sub(torch.sub(dimsA0, 1), offset0)
    dimB0 = torch.sub(torch.sub(dimsB0, 1), offset0)
    if torch.ge(dimA0, 0):
      sizeA0 = expandedSizes[dimA0]
    else:
      sizeA0 = 1
    if torch.ge(dimB0, 0):
      sizeB0 = c[dimB0]
    else:
      sizeB0 = 1
    if torch.ne(sizeA0, sizeB0):
      _7 = torch.ne(sizeA0, 1)
    else:
      _7 = False
    if _7:
      _8 = torch.ne(sizeB0, 1)
    else:
      _8 = False
    if _8:
      _9 = torch.format(_1, sizeA0, sizeB0, i0)
      ops.prim.RaiseException(torch.add("AssertionError: ", _9))
    else:
      pass
    if torch.eq(sizeA0, 1):
      _10 = sizeB0
    else:
      _10 = sizeA0
    _11 = torch.append(expandedSizes0, _10)
  return expandedSizes0

)=====")
+ std::string(R"=====(def broadcast_one_three(a: List[int],
    b: Any,
    c: List[int]) -> List[int]:
  _0 = "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
  dimsA = torch.len(a)
  dimsB = torch.len(c)
  ndim = ops.prim.max(dimsA, dimsB)
  expandedSizes = annotate(List[int], [])
  for i in range(ndim):
    offset = torch.sub(torch.sub(ndim, 1), i)
    dimA = torch.sub(torch.sub(dimsA, 1), offset)
    dimB = torch.sub(torch.sub(dimsB, 1), offset)
    if torch.ge(dimA, 0):
      sizeA = a[dimA]
    else:
      sizeA = 1
    if torch.ge(dimB, 0):
      sizeB = c[dimB]
    else:
      sizeB = 1
    if torch.ne(sizeA, sizeB):
      _1 = torch.ne(sizeA, 1)
    else:
      _1 = False
    if _1:
      _2 = torch.ne(sizeB, 1)
    else:
      _2 = False
    if _2:
      _3 = torch.add("AssertionError: ", torch.format(_0, sizeA, sizeB, i))
      ops.prim.RaiseException(_3)
    else:
      pass
    if torch.eq(sizeA, 1):
      _4 = sizeB
    else:
      _4 = sizeA
    _5 = torch.append(expandedSizes, _4)
  return expandedSizes

)=====")
+ std::string(R"=====(def broadcast_inplace(a: List[int],
    b: List[int]) -> List[int]:
  _0 = "The dims of tensor b ({}) must be less than or equal tothe dims of tensor a ({}) "
  _1 = "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
  dimsA = torch.len(a)
  dimsB = torch.len(b)
  if torch.gt(dimsB, dimsA):
    _2 = torch.add("AssertionError: ", torch.format(_0, dimsB, dimsA))
    ops.prim.RaiseException(_2)
  else:
    pass
  for dimA in range(dimsA):
    dimB = torch.add(torch.sub(dimsB, dimsA), dimA)
    sizeA = a[dimA]
    if torch.ge(dimB, 0):
      sizeB = b[dimB]
    else:
      sizeB = 1
    if torch.ne(sizeA, sizeB):
      _3 = torch.ne(sizeB, 1)
    else:
      _3 = False
    if _3:
      _4 = torch.format(_1, sizeA, sizeB, dimA)
      ops.prim.RaiseException(torch.add("AssertionError: ", _4))
    else:
      pass
  out = annotate(List[int], [])
  for _5 in range(torch.len(a)):
    elem = a[_5]
    _6 = torch.append(out, elem)
  return out

def nonzero_lower_bound(input: List[int]) -> List[int]:
  return [0, torch.len(input)]

def nonzero_upper_bound(input: List[int]) -> List[int]:
  numel = 1
  for _0 in range(torch.len(input)):
    elem = input[_0]
    numel = torch.mul(numel, elem)
  return [numel, torch.len(input)]

)=====")
;


const std::string& GetSerializedShapeFunctions() {
  return shape_funcs;
}


const OperatorMap<std::string>& GetShapeFunctionMappings() {
 static const OperatorMap<std::string> shape_mappings {
    {"aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)", "unary"},
    {"aten::rsub.Tensor(Tensor self, Scalar other, Scalar alpha=1) -> Tensor", "unary"},
    {"aten::dropout(Tensor input, float p, bool train) -> Tensor", "unary"},
    {"aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor", "adaptive_avg_pool2d"},
    {"prim::NumToTensor.Scalar(Scalar a) -> Tensor", "zero_dim_tensor"},
    {"prim::NumToTensor.bool(bool a) -> Tensor", "zero_dim_tensor"},
    {"aten::zeros(int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)", "unary"},
    {"aten::to.dtype(Tensor(a) self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))", "unary"},
    {"aten::arange(Scalar end, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)", "arange_end"},
    {"aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", "arange_start"},
    {"aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", "arange_start_step"},
    {"aten::squeeze(Tensor(a) self) -> Tensor(a)", "squeeze_nodim"},
    {"aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)", "squeeze"},
    {"aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)", "squeeze_dims"},
    {"aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)", "unsqueeze"},
    {"aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)", "slice"},
    {"aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)", "select"},
    {"aten::index_select(Tensor self, int dim, Tensor index) -> Tensor", "index_select"},
    {"aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor", "unary"},
    {"aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor", "unary"},
    {"aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor", "unary"},
    {"aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)", "unary"},
    {"aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor", "embedding"},
    {"aten::mm(Tensor self, Tensor mat2) -> Tensor", "mm"},
    {"aten::dot(Tensor self, Tensor tensor) -> Tensor", "dot"},
    {"aten::mv(Tensor self, Tensor vec) -> Tensor", "mv"},
    {"aten::matmul(Tensor self, Tensor other) -> Tensor", "matmul"},
    {"aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor", "linear"},
    {"aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor", "max_pool2d"},
    {"aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)", "max_pool2d_with_indices"},
    {"aten::t(Tensor(a) self) -> Tensor(a)", "t"},
    {"aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)", "transpose"},
    {"aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor", "conv1d"},
    {"aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor", "conv2d"},
    {"aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor", "batch_norm"},
    {"aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor", "conv3d"},
    {"aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight, int[]? bias_sizes, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)", "conv_backwards"},
    {"aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor", "conv_forwards"},
    {"aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor", "_conv_forwards"},
    {"aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor", "conv_transpose2d_input"},
    {"aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)", "flatten"},
    {"aten::cat(Tensor[] tensors, int dim=0) -> Tensor", "cat"},
    {"aten::stack(Tensor[] tensors, int dim=0) -> Tensor", "stack"},
    {"aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)", "permute"},
    {"aten::movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)", "movedim"},
    {"aten::view(Tensor(a) self, int[] size) -> Tensor(a)", "view"},
    {"aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)", "expand"},
    {"aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)", "expand_one_unused"},
    {"aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor", "sum_mean_dim"},
    {"aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor", "sum_mean_dim"},
    {"aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)", "max_dim"},
    {"aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor", "zero_dim_tensor"},
    {"aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor", "zero_dim_tensor"},
    {"aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor", "addmm"},
    {"aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)", "upsample_nearest2d"},
    {"aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor", "unary"},
    {"aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype) -> Tensor", "unary"},
    {"aten::dequantize(Tensor self) -> Tensor", "unary"},
    {"quantized::add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc", "broadcast"},
    {"aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor", "argmax"},
    {"aten::bmm(Tensor self, Tensor mat2) -> Tensor", "bmm"},
    {"aten::_shape_as_tensor(Tensor self) -> Tensor", "_shape_as_tensor"},
    {"aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)", "topk"},
    {"aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)", "nll_loss_forward"},
    {"aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)", "native_layer_norm"},
    {"aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)", "native_batch_norm"},
    {"aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)", "native_batch_norm"},
    {"aten::_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)", "native_batch_norm"},
    {"aten::_batch_norm_with_update(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor)", "_batch_norm_with_update"},
    {"aten::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, float label_smoothing=0.0) -> Tensor", "cross_entropy_loss"},
    {"aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor", "broadcast_three"},
    {"aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor", "broadcast_one_three"},
    {"aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)", "broadcast_inplace"},
  };

  return shape_mappings;
}

const OperatorMap<std::pair<std::string, std::string>>& GetBoundedShapeMappings() {
 static const OperatorMap<std::pair<std::string, std::string>> shape_mappings {
    {"aten::nonzero(Tensor self) -> (Tensor)", {"nonzero_lower_bound", "nonzero_upper_bound"}},
  };

  return shape_mappings;
}

// clang-format on

} // namespace torch::jit
