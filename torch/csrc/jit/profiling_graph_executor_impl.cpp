#include <torch/csrc/jit/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/guard_elimination.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/insert_guards.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>

namespace torch {
namespace jit {

thread_local bool profiling_mode = false;
bool& getProfilingMode() {
  return profiling_mode;
}

std::shared_ptr<Graph> ProfilingGraphExecutorImpl::prepareGraph(
    const std::shared_ptr<Graph>& graph,
    Stack& stack) {
  auto g = graph->copy();
  return g;
}

ProfilingGraphExecutorImpl::ProfilingGraphExecutorImpl(
    const std::shared_ptr<Graph>& graph,
    bool optimize)
    : GraphExecutorImplBase(graph, optimize), arg_spec_creator_(*this->graph) {}

ExecutionPlan ProfilingGraphExecutorImpl::getPlanFor(Stack& stack) {
  if (optimized_plan_) {
    return *optimized_plan_;
  }

  if (!pr_) {
    pr_ = ProfilingRecord::instrumentGraph(prepareGraph(graph, stack));
    profiling_plan_ = ExecutionPlan(pr_->profiled_graph_);
    // fall-through
  }

  if (!pr_->ready()) {
    return *profiling_plan_;
  }
  // copy already has differentiableGraphs
  auto copy = pr_->graph()->copy();
  // insert bailouts
  InsertGuards(copy);
  EliminateRedundantGuards(copy);
  InsertBailOuts(copy);
  // TODO: this runs specializeAutogradZero ??
  runRequiredPasses(copy);
  if (needsGradient(copy)) {
    auto diff_nodes = CreateAutodiffSubgraphs(
        copy, getAutodiffSubgraphInlining() ? autodiffSubgraphNodeThreshold : 1);
    for (Node* dnode : diff_nodes) {
      auto diff_graph = std::move(dnode->g(attr::Subgraph));
      Gradient gradient = differentiate(diff_graph);
      // do not optimize DifferentiableGraphs, since
      // ideally they will be profiled and then optimized separetely
      // when their corresponding DifferentiableGraphOp is called
      packGradient(gradient, dnode);
    }
    InlineAutodiffSubgraphs(
        copy, getAutodiffSubgraphInlining() ? autodiffSubgraphInlineThreshold : 1);
  }
  ConstantPropagation(copy);
  runOptimization(copy);
  runNondiffOptimization(copy);
  EliminateDeadCode(copy);
  // cache
  optimized_plan_ = ExecutionPlan(copy);
  return *optimized_plan_;
}


GraphExecutorState ProfilingGraphExecutorImpl::getDebugState() {
  AT_ERROR("not supported");
}

} // namespace jit
} // namespace torch
