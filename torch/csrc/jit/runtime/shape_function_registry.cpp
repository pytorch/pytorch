
/**
 * @generated
 * This is an auto-generated file. Please do not modify it by hand.
 * To re-generate, please run:
 * cd ~/pytorch && python tools/codegen/decompositions/gen_jit_shape_functions.py
 */
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/shape_function_registry.h>

namespace torch {
namespace jit {


const std::string decomp_funcs =
R"=====("
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

def arange_end(end: number,
    inp0: Any,
    inp1: Any,
    inp2: Any,
    inp3: Any) -> List[int]:
  if torch.ge(end, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  return [int(torch.ceil(end))]

def arange_start(start: number,
    end: number,
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

def arange_start_step(start: number,
    end: number,
    step: number,
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

def slice(self: List[int],
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
  len = torch.sub(end_val1, start_val2)
  out = annotate(List[int], [])
  for _2 in range(torch.len(self)):
    elem = self[_2]
    _3 = torch.append(out, elem)
  _4 = torch.floordiv(torch.sub(torch.add(len, step), 1), step)
  _5 = torch._set_item(out, dim0, _4)
  return out

def select(self: List[int],
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

def index_select(self: List[int],
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

def embedding(weight: List[int],
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

")=====" R"=====("def dot(self: List[int],
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

def matmul(tensor1: List[int],
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

")====="
R"=====("
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

def arange_end(end: number,
    inp0: Any,
    inp1: Any,
    inp2: Any,
    inp3: Any) -> List[int]:
  if torch.ge(end, 0):
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  return [int(torch.ceil(end))]

def arange_start(start: number,
    end: number,
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

def arange_start_step(start: number,
    end: number,
    step: number,
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

def slice(self: List[int],
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
  len = torch.sub(end_val1, start_val2)
  out = annotate(List[int], [])
  for _2 in range(torch.len(self)):
    elem = self[_2]
    _3 = torch.append(out, elem)
  _4 = torch.floordiv(torch.sub(torch.add(len, step), 1), step)
  _5 = torch._set_item(out, dim0, _4)
  return out

def select(self: List[int],
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

def index_select(self: List[int],
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

def embedding(weight: List[int],
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

")=====" R"=====("def dot(self: List[int],
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

def matmul(tensor1: List[int],
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

")=====";


const std::string& GetSerializedDecompositions() {
  return decomp_funcs;
}

const OperatorMap<std::string>& GetDecompositionMapping() {
  // clang-format off
 static const OperatorMap<std::string> decomposition_mapping {
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
    {"aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)", "flatten"},
    {"aten::cat(Tensor[] tensors, int dim=0) -> Tensor", "cat"},
    {"aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)", "permute"},
    {"aten::view(Tensor(a) self, int[] size) -> Tensor(a)", "view"},
    {"aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)", "expand"},
    {"aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)", "expand_one_unused"},
    {"aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor", "mean_dim"},
    {"aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor", "mean_dim"},
    {"aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)", "max_dim"},
    {"aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor", "zero_dim_tensor"},
    {"aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor", "zero_dim_tensor"},
    {"aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor", "addmm"},
    {"aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)", "upsample_nearest2d"},
    {"aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor", "unary"},
    {"aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype) -> Tensor", "unary"},
    {"aten::dequantize(Tensor self) -> Tensor", "unary"},
    {"quantized::add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc", "broadcast"},
  };
  // clang-format on

  return decomposition_mapping;
}

} // namespace jit
} // namespace torch
