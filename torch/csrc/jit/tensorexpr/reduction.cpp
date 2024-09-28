
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <utility>

namespace torch::jit::tensorexpr {

ExprHandle Reducer::operator()(
    const BufHandle& result_buf,
    ExprHandle body,
    const std::vector<ExprHandle>& output,
    const std::vector<VarHandle>& inner) const {
  return ReduceOp::make(
      complete(result_buf, interaction_, std::move(body), output, inner),
      inner,
      *this);
}

ReduceOpPtr Reducer::operator()(
    const BufPtr& result_buf,
    ExprPtr body,
    const std::vector<ExprPtr>& output,
    const std::vector<VarPtr>& inner) const {
  return alloc<ReduceOp>(
      complete(
          result_buf, interaction_, ExprHandle(std::move(body)), output, inner),
      inner,
      *this);
}

ExprHandle Reducer::operator()(
    const BufHandle& result_buf,
    BufHandle acc_buf,
    const ExprHandle& body,
    const std::vector<ExprHandle>& output,
    const std::vector<VarHandle>& inner) const {
  return ReduceOp::make(
      complete(result_buf, interaction_, body, output, inner),
      inner,
      result_buf,
      std::move(acc_buf),
      body,
      *this);
}

ExprHandle ReduceOp::make(
    ExprHandle body,
    const std::vector<VarHandle>& reduce_args,
    const Reducer& reducer) {
  return ExprHandle(alloc<ReduceOp>(
      body.node(), VarHandleVectorToVarVector(reduce_args), reducer));
}

ExprHandle ReduceOp::make(
    ExprHandle body,
    const std::vector<VarHandle>& reduce_args,
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

} // namespace torch::jit::tensorexpr
