#include <torch/csrc/jit/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/guard_elimination.h>
#include <torch/csrc/jit/passes/insert_guards.h>

namespace torch {
namespace jit {

thread_local bool profiling_mode = false;
bool& getProfilingMode() {
  return profiling_mode;
}

static std::shared_ptr<Graph> prepareGraph(const std::shared_ptr<Graph>& graph) {
  auto g = graph->copy();
  runRequiredPasses(g);
  return g;
}

ProfilingGraphExecutorImpl::ProfilingGraphExecutorImpl(const std::shared_ptr<Graph>& graph, bool optimize)
: GraphExecutorImplBase(graph, optimize),
  pr_(ProfilingRecord::instrumentGraph(prepareGraph(graph))),
  profiling_plan_(pr_->graph()) {}

ExecutionPlan ProfilingGraphExecutorImpl::getPlanFor(Stack& stack) {
  if (optimized_plan_) {
    return *optimized_plan_;
  }
  if (!pr_->ready()) {
    return profiling_plan_;
  }

  auto copy = pr_->graph()->copy();
  std::cout << "before InsertGuards\n";
  copy->dump();
  InsertGuards(copy);
  EliminateGuards(copy);
  std::cout << "before InsertBailOuts\n";
  copy->dump();
  InsertBailOuts(copy);
  optimized_plan_ = ExecutionPlan(copy);
  std::cout << "optimized graph = \n";
  copy->dump();
  return *optimized_plan_;
}


GraphExecutorState ProfilingGraphExecutorImpl::getDebugState() {
  AT_ERROR("not supported");
}

} // namespace jit
} // namespace torch
