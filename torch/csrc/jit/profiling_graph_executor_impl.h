#pragma once
#include <torch/csrc/jit/graph_executor_impl.h>

namespace torch {
namespace jit {

static const size_t MAX_BAILOUT_DEPTH = 1;

struct ProfilingGraphExecutorImpl : public GraphExecutorImplBase {
  ProfilingGraphExecutorImpl(const std::shared_ptr<Graph>& graph);

  ExecutionPlan getPlanFor(Stack& stack) override;
  GraphExecutorState getDebugState() override;
  ~ProfilingGraphExecutorImpl() override = default;

 private:
  void runProfilingInsensitiveOptimizations(std::shared_ptr<Graph>& graph);
  void runProfilingOptimizations(std::shared_ptr<Graph>& graph);
  std::unique_ptr<ProfilingRecord> pr_;
  c10::optional<ExecutionPlan>
      profiling_plan_; // plan to run in order to profiling the code
  c10::optional<ExecutionPlan> optimized_plan_;
  int64_t bailout_depth_;
};

} // namespace jit
} // namespace torch
