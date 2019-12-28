#include "torch/csrc/jit/compiler/include/tensor.h"
#include "torch/csrc/jit/compiler/include/schedule.h"

namespace torch {
namespace jit {
namespace compiler {

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
  CHECK(expr_node_ != nullptr);
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

} // namespace compiler
} // namespace jit
} // namespace torch
