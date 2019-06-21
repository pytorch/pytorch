#pragma once
#include <torch/csrc/jit/graph_executor_impl.h>

namespace torch {
namespace jit {

struct ProfilingGraphExecutorImpl : public GraphExecutorImplBase {
  ProfilingGraphExecutorImpl(const std::shared_ptr<Graph>& graph, bool optimize);

  ExecutionPlan getPlanFor(Stack& stack) override;
  GraphExecutorState getDebugState() override;
  ~ProfilingGraphExecutorImpl() override = default;

 private:
  std::shared_ptr<Graph> prepareGraph(
      const std::shared_ptr<Graph>& graph,
      Stack& stack);
  std::unique_ptr<ProfilingRecord> pr_;
  c10::optional<ExecutionPlan>
      profiling_plan_; // plan to run in order to profiling the code
  c10::optional<ExecutionPlan> optimized_plan_;
  ArgumentSpecCreator arg_spec_creator_;
};

} // namespace jit
} // namespace torch
