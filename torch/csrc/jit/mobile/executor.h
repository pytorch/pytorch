#pragma once

#include <torch/csrc/jit/runtime/graph_executor.h>

namespace torch {
namespace jit {
namespace mobile {

struct Code;

class EdgeExecutionPlan : public torch::jit::ExecutionPlan {
 public:
  EdgeExecutionPlan(const Code& code) : ExecutionPlan(), code_(code) {}

  bool isEdgeExecutionPlan() const override {
    return true;
  }

  const Code& getCode() const {
    return code_;
  }

 private:
  const Code& code_;
};

const EdgeExecutionPlan& toEdgeExecutionPlan(const ExecutionPlan& plan);

class EdgeExecutor : public torch::jit::Executor {
 public:
  EdgeExecutor(const Code& code) : plan_(code) {}
  const ExecutionPlan& getPlanFor(Stack& inputs, size_t remaining_bailout_depth)
      override {
    TORCH_INTERNAL_ASSERT(remaining_bailout_depth == 0);
    return plan_;
  }

  GraphExecutorState getDebugState() override {
    TORCH_INTERNAL_ASSERT(
        false, "getDebugState not supported for EdgeExecutor.");
  }
  void debugFlushCompilationCache() override {}

 private:
  EdgeExecutionPlan plan_;
};

} // namespace mobile
} // namespace jit
} // namespace torch
