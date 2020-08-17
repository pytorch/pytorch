#pragma once
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

namespace torch {
namespace jit {

Function* createFallbackPathFunction(Block* b, const std::string& function_name);
TORCH_API Node* insertFallbackFunctionCall(Graph* graph, Function* func, ArrayRef<Value*> values);

struct ProfilingGraphExecutorImpl : public GraphExecutorImplBase {
  ProfilingGraphExecutorImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name);

  ExecutionPlan getPlanFor(Stack& stack, size_t remaining_bailout_depth)
      override;
  GraphExecutorState getDebugState() override;
  ~ProfilingGraphExecutorImpl() override = default;

 private:
  void runProfilingInsensitiveOptimizations(std::shared_ptr<Graph>& graph);
  void runProfilingOptimizations(std::shared_ptr<Graph>& graph);
  void registerFallbackFunctions(Block* b);
  std::unique_ptr<ProfilingRecord> pr_;
  c10::optional<ExecutionPlan>
      profiling_plan_; // plan to run in order to profiling the code
  c10::optional<ExecutionPlan> optimized_plan_;
  std::vector<std::unique_ptr<Function>> fallback_functions_;
};

} // namespace jit
} // namespace torch
