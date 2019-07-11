#pragma once
#include <torch/csrc/jit/graph_executor_impl.h>

namespace torch {
namespace jit {

struct ProfilingGraphExecutorImpl : public GraphExecutorImplBase {
  using GraphExecutorImplBase::GraphExecutorImplBase;

  void run(Stack& stack) override;
  GraphExecutorState getDebugState() override;
  ~ProfilingGraphExecutorImpl() override = default;

 private:
  std::unique_ptr<ProfilingRecord> pr_;
  std::unique_ptr<ExecutionPlan> exec_plan_;
};

} // namespace jit
} // namespace torch
