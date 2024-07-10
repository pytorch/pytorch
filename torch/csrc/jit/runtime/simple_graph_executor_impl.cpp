#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>

#include <c10/util/Optional.h>
#include <torch/csrc/jit/runtime/simple_graph_executor_impl.h>
#include <mutex>

namespace torch::jit {

SimpleGraphExecutorImpl::SimpleGraphExecutorImpl(
    const std::shared_ptr<Graph>& graph,
    std::string function_name)
    : GraphExecutorImplBase(graph, std::move(function_name)) {}

const ExecutionPlan& SimpleGraphExecutorImpl::getPlanFor(
    Stack& stack,
    std::optional<size_t> remaining_bailout_depth) {
  std::lock_guard<std::mutex> lock(compile_mutex);

  // IMPORTANT: This is a hot path of calling a torchscript function. Try not to
  // add any code above this.
  if (execution_plan_) {
    return *execution_plan_;
  }
  auto copy = graph->copy();
  runNooptPassPipeline(copy);
  execution_plan_ = ExecutionPlan(copy, function_name_);

  return *execution_plan_;
}

GraphExecutorState SimpleGraphExecutorImpl::getDebugState() {
  GraphExecutorState state;
  TORCH_INTERNAL_ASSERT(execution_plan_);
  state.graph = execution_plan_->graph.get();
  auto opt_plan = *execution_plan_;
  state.execution_plans.emplace(ArgumentSpec{0, 0}, opt_plan);
  return state;
}

} // namespace torch::jit
