#pragma once
#include <torch/csrc/jit/runtime/graph_executor_impl.h>
#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

struct TORCH_API ProfilingGraphExecutorImpl : public GraphExecutorImplBase {
  ProfilingGraphExecutorImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name);

  const ExecutionPlan& getPlanFor(Stack& stack, size_t remaining_bailout_depth)
      override;
  GraphExecutorState getDebugState() override;
  ~ProfilingGraphExecutorImpl() override = default;

  void debugFlushCompilationCache() {
    std::lock_guard<std::mutex> lock(compile_mutex);
    pr_.reset();
    fallback_plan_.reset();
    profiling_plan_.reset();
    optimized_plan_.reset();
    // prevent memory leaks
    fallback_functions_.clear();
    remaining_bailout_depth_.reset();
  }

  bool isOptimized() const override {
    return optimized_plan_.has_value();
  }

 private:
  const ExecutionPlan& getOptimizedPlanFor(
      Stack& stack,
      size_t remaining_bailout_depth);
  void runProfilingInsensitiveOptimizations(std::shared_ptr<Graph>& graph);
  void runProfilingOptimizations(
      std::shared_ptr<Graph>& graph,
      size_t remaining_depth);
  void replaceFallbackGraphWithFallbackFunction(Block* b);
  std::unique_ptr<ProfilingRecord> pr_;
  c10::optional<ExecutionPlan>
      profiling_plan_; // plan to run in order to profiling the code
  c10::optional<ExecutionPlan> optimized_plan_;
  // this plan is used if getGraphExecutorOptimize is unset
  c10::optional<ExecutionPlan> fallback_plan_;
  // fallback functions are inserted for tensorexpr fusion groups
  // and by specialize_autogradzero. Whenever, at runtime, input
  // tensor don't match profiled properties, fallback functions are called
  // They are the deoptimized version of the logic in fusion groups
  // and/or autograd.
  // The fallback functions are owned by a GraphExecutor instance
  // They only exist in the optimized graph which is a private property
  // of the GraphExecutor and only shared with InterpreterState
  std::vector<std::unique_ptr<Function>> fallback_functions_;
  c10::optional<size_t> remaining_bailout_depth_;
};

} // namespace jit
} // namespace torch
