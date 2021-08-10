
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

ReduceOp* Reducer::operator()(
    Buf* result_buf,
    ExprHandle body,
    const std::vector<Expr*>& output,
    const std::vector<Var*>& inner) const {
  return new ReduceOp(
      complete(result_buf, interaction_, body, output, inner), inner, *this);
}

ReduceOp* Reducer::operator()(
    Buf* result_buf,
    Expr* body,
    const std::vector<Expr*>& output,
    const std::vector<Var*>& inner) const {
  return new ReduceOp(
      complete(result_buf, interaction_, ExprHandle(body), output, inner),
      inner,
      *this);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
