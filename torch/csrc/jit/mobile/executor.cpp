#include <c10/util/Exception.h>
#include <torch/csrc/jit/mobile/executor.h>

namespace torch {
namespace jit {
namespace mobile {

const EdgeExecutionPlan& toEdgeExecutionPlan(const ExecutionPlan& plan) {
  TORCH_INTERNAL_ASSERT(plan.isEdgeExecutionPlan());
  return static_cast<const EdgeExecutionPlan&>(plan);
}

} // namespace mobile
} // namespace jit
} // namespace torch
