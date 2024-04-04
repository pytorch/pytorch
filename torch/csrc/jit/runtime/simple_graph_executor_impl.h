#pragma once
#include <c10/util/Flags.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

namespace torch::jit {

struct TORCH_API SimpleGraphExecutorImpl : public GraphExecutorImplBase {
  SimpleGraphExecutorImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name);

  const ExecutionPlan& getPlanFor(
      Stack& stack,
      c10::optional<size_t> remaining_bailout_depth) override;
  GraphExecutorState getDebugState() override;
  ~SimpleGraphExecutorImpl() override = default;

 private:
  c10::optional<ExecutionPlan> execution_plan_;
};

} // namespace torch::jit
