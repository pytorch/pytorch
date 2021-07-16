
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

ReduceOp* Reducer::operator()(
    const Buf* result_buf,
    ExprHandle body,
    const std::vector<const Expr*>& output,
    const std::vector<const Var*>& inner) const {
  return new ReduceOp(
      complete(result_buf, interaction_, body, output, inner), inner, *this);
}

ReduceOp* Reducer::operator()(
    const Buf* result_buf,
    const Expr* body,
    const std::vector<const Expr*>& output,
    const std::vector<const Var*>& inner) const {
  return new ReduceOp(
      complete(result_buf, interaction_, ExprHandle(body), output, inner),
      inner,
      *this);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
