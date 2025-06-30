#pragma once
#include <c10/util/Flags.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

TORCH_DECLARE_bool(torch_jit_static_then_dynamic);

TORCH_DECLARE_bool(torch_jit_always_dynamic);

C10_DECLARE_bool(torch_jit_release_profiling_graph_after_optimization);
C10_DECLARE_int32(torch_jit_release_profiling_graph_delay_in_seconds);
C10_DECLARE_int64(torch_jit_num_profiled_runs);
C10_DECLARE_int64(torch_jit_bailout_depth);

namespace torch::jit {

TORCH_API void runNooptPassPipeline(std::shared_ptr<Graph>& graph);

struct TORCH_API ProfilingGraphExecutorImpl : public GraphExecutorImplBase {
  ProfilingGraphExecutorImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name);

  const ExecutionPlan& getPlanFor(
      Stack& stack,
      std::optional<size_t> remaining_bailout_depth) override;
  GraphExecutorState getDebugState() override;
  ~ProfilingGraphExecutorImpl() override = default;

  void debugFlushCompilationCache();

  bool isOptimized() const override {
    return optimized_plan_.has_value();
  }

 private:
  const ExecutionPlan& getOptimizedPlanFor(
      Stack& stack,
      std::optional<size_t> remaining_bailout_depth);
  void runProfilingInsensitiveOptimizations(std::shared_ptr<Graph>& graph);
  void runProfilingOptimizations(
      std::shared_ptr<Graph>& graph,
      size_t remaining_depth);
  void replaceFallbackGraphWithFallbackFunction(Block* b);
  FusionBehavior getCurrentBehavior(size_t remaining_depth);
  size_t getInstantiatedBailoutDepth();
  void runNoGradOptimizations(
      std::shared_ptr<Graph>& graph,
      size_t remaining_bailout_depth);
  void runFinalOptimizations(std::shared_ptr<Graph>& graph);

  void clearTheGraphCompilationIntermediateGraphs();

  std::unique_ptr<ProfilingRecord> pr_;
  std::optional<ExecutionPlan>
      profiling_plan_; // plan to run in order to profiling the code
  std::optional<ExecutionPlan> optimized_plan_;
  FusionStrategy fusion_strategy_;

  // this plan is used if getGraphExecutorOptimize is unset
  std::optional<ExecutionPlan> fallback_plan_;
  // fallback functions are inserted for tensorexpr fusion groups
  // and by specialize_autogradzero. Whenever, at runtime, input
  // tensor don't match profiled properties, fallback functions are called
  // They are the deoptimized version of the logic in fusion groups
  // and/or autograd.
  // The fallback functions are owned by a GraphExecutor instance
  // They only exist in the optimized graph which is a private property
  // of the GraphExecutor and only shared with InterpreterState
  std::vector<std::unique_ptr<Function>> fallback_functions_;
  std::optional<size_t> remaining_bailout_depth_;
  // The time the optimized_plan_ is created.
  int32_t time_optimized_plan_created_ = 0;
  // Has the extra memory used by the graph for profiling is released?
  bool is_graph_extra_memory_released_ = false;
};

} // namespace torch::jit
