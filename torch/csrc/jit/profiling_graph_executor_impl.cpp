#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/canonicalize_ops.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/guard_elimination.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/insert_guards.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/profiling_graph_executor_impl.h>

namespace torch {
namespace jit {

static bool profiling_mode = false;
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
    const std::shared_ptr<Graph>& graph)
    : GraphExecutorImplBase(graph), arg_spec_creator_(*this->graph) {}

ExecutionPlan ProfilingGraphExecutorImpl::getPlanFor(Stack& stack) {
  if (optimized_plan_) {
    return *optimized_plan_;
  }

  if (!pr_) {
    pr_ = ProfilingRecord::instrumentGraph(prepareGraph(graph, stack));
    auto copy = pr_->graph()->copy();
    LowerGradOf(*copy);
    RemoveExpands(copy);
    CanonicalizeOps(copy);
    EliminateDeadCode(copy);
    profiling_plan_ = ExecutionPlan(copy);
    // fall-through
  }

  if (!pr_->ready()) {
    return *profiling_plan_;
  }
  // copy already has differentiableGraphs
  auto copy = pr_->graph()->copy();
  // insert bailouts
  InsertGuards(copy);
  runRequiredPasses(copy);
  EliminateRedundantGuards(copy);
  InsertBailOuts(copy);
  // TODO: this runs specializeAutogradZero ??
  GRAPH_DUMP("After InsertBailOuts: ", copy);
  runRequiredPasses(copy);
  if (needsGradient(copy)) {
    auto diff_nodes = CreateAutodiffSubgraphs(
        copy,
        getAutodiffSubgraphInlining() ? autodiffSubgraphNodeThreshold : 1);
    for (Node *dnode : diff_nodes) {
      auto diff_graph = std::move(dnode->g(attr::Subgraph));
      Gradient gradient = differentiate(diff_graph);
      runOptimization(gradient.f);
      // run non diff optimization on the forward graph
      runNondiffOptimization(gradient.f);
      packGradient(gradient, dnode);
    }
    InlineAutodiffSubgraphs(copy, getAutodiffSubgraphInlining()
                                      ? autodiffSubgraphInlineThreshold
                                      : 1);
  }
  GRAPH_DUMP("InlineAutodiffSubgraphs: ", copy);
  ConstantPropagation(copy);
  runOptimization(copy);
  runNondiffOptimization(copy);
  EliminateDeadCode(copy);
  // cache
  optimized_plan_ = ExecutionPlan(copy);
  GRAPH_DUMP("ExecutionPlan: ", copy);
  return *optimized_plan_;
}


GraphExecutorState ProfilingGraphExecutorImpl::getDebugState() {
  GraphExecutorState state;
  TORCH_INTERNAL_ASSERT(optimized_plan_);
  auto opt_plan = *optimized_plan_;
  state.execution_plans.emplace(ArgumentSpec{0, 0}, opt_plan);
  return state;
}

} // namespace jit
} // namespace torch
