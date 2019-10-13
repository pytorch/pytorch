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

static std::atomic<bool> profiling_mode{false};
std::atomic<bool>& getProfilingMode() {
  return profiling_mode;
}

std::shared_ptr<Graph> ProfilingGraphExecutorImpl::prepareGraph(
    const std::shared_ptr<Graph>& graph,
    Stack& stack) {
  auto g = graph->copy();
  ArgumentSpec spec =
      arg_spec_creator_.create(autograd::GradMode::is_enabled(), stack);
  arg_spec_creator_.specializeTypes(*g, spec);
  runRequiredPasses(g);
  PropagateRequiresGrad(g);
  ConstantPropagation(g);
  if (needsGradient(g)) {
    auto diff_nodes = CreateAutodiffSubgraphs(
        g, getAutodiffSubgraphInlining() ? autodiffSubgraphNodeThreshold : 1);
    for (Node* dnode : diff_nodes) {
      auto diff_graph = std::move(dnode->g(attr::Subgraph));
      Gradient gradient = differentiate(diff_graph);
      // do not optimize DifferentiableGraphs, since
      // ideally they will be profiled and then optimized separetely
      // when their corresponding DifferentiableGraphOp is called
      packGradient(gradient, dnode);
    }
    InlineAutodiffSubgraphs(
        g, getAutodiffSubgraphInlining() ? autodiffSubgraphInlineThreshold : 1);
  }
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
  if (!getGraphExecutorOptimize()) {
    runRequiredPasses(copy);
    optimized_plan_ = ExecutionPlan(copy);
    return *optimized_plan_;
  }

  // insert bailouts
  InsertGuards(copy);
  // get rid of autograd specific ops
  // we can probably make guard_elimination.cpp
  // to handle these ops
  specializeAutogradZero(*copy);
  // hoist out GradOf blocks
  // otherwise we will need to teach
  // liveness and buildBailOut graphs
  // about them
  LowerGradOf(*copy);
  // constant fold into ConstantChunk
  CanonicalizeOps(copy);
  EliminateRedundantGuards(copy);
  InsertBailOuts(copy);
  // TODO: this runs specializeAutogradZero ??
  GRAPH_DUMP("After InsertBailOuts: ", copy);
  runRequiredPasses(copy);
  ConstantPropagation(copy);
  runOptimization(copy);
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
  } else {
    runNondiffOptimization(copy);
  }
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
