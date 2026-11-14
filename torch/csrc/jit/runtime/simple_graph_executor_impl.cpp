#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>

#include <torch/csrc/jit/runtime/simple_graph_executor_impl.h>
#include <mutex>
#include <optional>

// clang-format off
C10_DEFINE_bool(
    torch_jit_release_simple_executor_graph,
    false,
    "After SimpleGraphExecutorImpl compiles its execution plan, release the unoptimized graph the plan was built from. "
    "The plan owns its own copy, so the original is only needed until compilation finishes. Inference-only memory saving; "
    "any future code path that rebuilds a plan from the original graph will fail.")
// clang-format on

namespace torch::jit {

SimpleGraphExecutorImpl::SimpleGraphExecutorImpl(
    const std::shared_ptr<Graph>& graph,
    std::string function_name)
    : GraphExecutorImplBase(graph, std::move(function_name)) {}

const ExecutionPlan& SimpleGraphExecutorImpl::buildPlan() {
  TORCH_CHECK(
      graph,
      "SimpleGraphExecutorImpl: graph was released before a plan was built");
  auto copy = graph->copy();
  runNooptPassPipeline(copy);
  execution_plan_ = ExecutionPlan(copy, function_name_);
  if (FLAGS_torch_jit_release_simple_executor_graph) {
    graph.reset();
  }

  return *execution_plan_;
}

const ExecutionPlan& SimpleGraphExecutorImpl::getPlanFor(
    Stack& stack,
    std::optional<size_t> remaining_bailout_depth) {
  std::lock_guard<std::mutex> lock(compile_mutex);

  // IMPORTANT: This is a hot path of calling a torchscript function. Try not to
  // add any code above this.
  if (execution_plan_) {
    return *execution_plan_;
  }
  return buildPlan();
}

const ExecutionPlan& SimpleGraphExecutorImpl::getInputIndependentPlan() {
  std::lock_guard<std::mutex> lock(compile_mutex);
  return getInputIndependentPlanImpl();
}

const ExecutionPlan& SimpleGraphExecutorImpl::getInputIndependentPlanImpl() {
  if (execution_plan_) {
    return *execution_plan_;
  }
  return buildPlan();
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
