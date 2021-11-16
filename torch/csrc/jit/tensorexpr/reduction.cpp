
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

ReduceOpPtr Reducer::operator()(
    BufPtr result_buf,
    ExprHandle body,
    const std::vector<ExprPtr>& output,
    const std::vector<VarPtr>& inner) const {
  return alloc<ReduceOp>(
      complete(result_buf, interaction_, body, output, inner), inner, *this);
}

ReduceOpPtr Reducer::operator()(
    BufPtr result_buf,
    ExprPtr body,
    const std::vector<ExprPtr>& output,
    const std::vector<VarPtr>& inner) const {
  return alloc<ReduceOp>(
      complete(result_buf, interaction_, ExprHandle(body), output, inner),
      inner,
      *this);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
