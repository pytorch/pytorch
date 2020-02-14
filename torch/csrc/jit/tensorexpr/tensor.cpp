#include "torch/csrc/jit/tensorexpr/tensor.h"
#include "torch/csrc/jit/tensorexpr/schedule.h"

namespace torch {
namespace jit {
namespace tensorexpr {

using schedule::TensorExprNode;
// using schedule::ScheduleNode;

void TensorOperationNode::SplitWithTail(
    const Var& loop_var,
    int factor,
    bool factor_on_inner,
    Var* outer_var,
    Var* inner_var,
    Var* tail_var,
    TensorOperation* tail_op) {
  check_expr_node();
  schedule::ScheduleNode* schedule = expr_node_->schedule();
  schedule::TensorExprNode* tail_expr_node = nullptr;
  schedule->SplitWithTail(
      expr_node_,
      loop_var,
      factor,
      factor_on_inner,
      outer_var,
      inner_var,
      tail_var,
      &tail_expr_node);
  if (!tail_expr_node) {
    *tail_op = TensorOperation::make(tail_expr_node);
  }
}

void TensorOperationNode::SplitWithMask(
    const Var& loop_var,
    int factor,
    bool factor_on_inner,
    Var* outer_var,
    Var* inner_var) {
  check_expr_node();
  schedule::ScheduleNode* schedule = expr_node_->schedule();
  schedule::TensorExprNode* tail_expr_node = nullptr;
  schedule->SplitWithMask(
      expr_node_, loop_var, factor, factor_on_inner, outer_var, inner_var);
}

void TensorOperationNode::GPUExecConfig(
    const std::vector<Var>& blockIdx,
    const std::vector<Var>& threadIdx) {
  check_expr_node();
  schedule::ScheduleNode* schedule = expr_node_->schedule();
  schedule->GPUExecConfig(expr_node_, blockIdx, threadIdx);
}

void TensorOperationNode::ComputeInline() {
  check_expr_node();
  schedule::ScheduleNode* schedule = expr_node_->schedule();
  schedule->ComputeInline(expr_node_);
}

void TensorOperationNode::check_expr_node() {
  if (expr_node_ == nullptr) {
    throw std::runtime_error(
        "expr_node in this tensor is null. It is likely that no schedule is attached.");
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
