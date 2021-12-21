#pragma once

#include <ATen/core/jit_type.h>

namespace torch {
namespace jit {
// Dtype constraints are not constrained in compilation. Therefore, we map
// all tensor subclasses with different dtypes to a same underlying
// Tensor. But, we give warning about possible dtype change whenever user
// uses any of the tensor subclasses such as LongTensor.
//
// Technically "number" is not a python type but we need it when
// parsing serialized methods that use implicit conversions to Scalar
#define FORALL_JIT_BASE_TYPES(_) \
  _(Tensor, Tensor)              \
  _(LongTensor, Tensor)          \
  _(DoubleTensor, Tensor)        \
  _(FloatTensor, Tensor)         \
  _(IntTensor, Tensor)           \
  _(ShortTensor, Tensor)         \
  _(HalfTensor, Tensor)          \
  _(CharTensor, Tensor)          \
  _(ByteTensor, Tensor)          \
  _(BoolTensor, Tensor)          \
  _(int, Int)                    \
  _(float, Float)                \
  _(bool, Bool)                  \
  _(complex, Complex)            \
  _(str, String)                 \
  _(Device, DeviceObj)           \
  _(Stream, StreamObj)           \
  _(number, Number)              \
  _(None, None)                  \
  _(NoneType, None)              \
  _(Any, Any)                    \
  _(Capsule, Capsule)            \
  _(list, AnyList)               \
  _(tuple, AnyTuple)

const std::unordered_map<std::string, c10::TypePtr>& string_to_type_lut();

} // namespace jit
} // namespace torch
