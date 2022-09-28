
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

ExprHandle Reducer::operator()(
    BufHandle result_buf,
    ExprHandle body,
    const std::vector<ExprHandle>& output,
    const std::vector<VarHandle>& inner) const {
  return ReduceOp::make(
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

ExprHandle Reducer::operator()(
    BufHandle result_buf,
    BufHandle acc_buf,
    ExprHandle body,
    const std::vector<ExprHandle>& output,
    const std::vector<VarHandle>& inner) const {
  return ReduceOp::make(
      complete(result_buf, interaction_, body, output, inner),
      inner,
      result_buf,
      acc_buf,
      body,
      *this);
}

ExprHandle ReduceOp::make(
    ExprHandle body,
    std::vector<VarHandle> reduce_args,
    const Reducer& reducer) {
  return ExprHandle(alloc<ReduceOp>(
      body.node(), VarHandleVectorToVarVector(reduce_args), reducer));
}

ExprHandle ReduceOp::make(
    ExprHandle body,
    std::vector<VarHandle> reduce_args,
    BufHandle result_buf,
    BufHandle acc_buf,
    ExprHandle ri_operand,
    const Reducer& reducer) {
  return ExprHandle(alloc<ReduceOp>(
      body.node(),
      VarHandleVectorToVarVector(reduce_args),
      result_buf.node(),
      acc_buf.node(),
      ri_operand.node(),
      reducer));
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
