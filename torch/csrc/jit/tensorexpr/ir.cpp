#include "torch/csrc/jit/tensorexpr/ir.h"

#include "torch/csrc/jit/tensorexpr/buffer.h"

namespace torch {
namespace jit {
namespace tensorexpr {

static Dtype ChooseDtype(const Dtype& buffer_dtype, const Dtype& index_dtype) {
  return Dtype(buffer_dtype, index_dtype.lanes());
}

Load::Load(const Buffer& buffer, const Expr& index, const Expr& mask)
    : Load(
          ChooseDtype(buffer.dtype(), index.dtype()),
          buffer.data(),
          index,
          mask) {}

Load::Load(
    Dtype dtype,
    const Var& base_handle,
    const Expr& index,
    const Expr& mask)
    : ExprNodeBase(dtype),
      base_handle_(base_handle),
      index_(index),
      mask_(mask) {
  CHECK_EQ(base_handle_.dtype(), kHandle);
  CHECK_EQ(index.dtype().lanes(), mask.dtype().lanes());
  CHECK_EQ(index.dtype().scalar_type(), kInt32);
}

Store::Store(
    const Buffer& buffer,
    const Expr& index,
    const Expr& value,
    const Expr& mask)
    : Store(buffer.data(), index, value, mask) {
  CHECK_EQ(buffer.dtype().scalar_type(), value.dtype().scalar_type());
  CHECK_EQ(buffer.dtype().scalar_type(), value.dtype().scalar_type());
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
