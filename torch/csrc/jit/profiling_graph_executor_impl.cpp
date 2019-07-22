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
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/canonicalize_ops.h>


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
  std::cout << "in ProfilingGraphExecutorImpl::getPlanFor " << this << std::endl;
  if (optimized_plan_) {
    std::cout << "returning optimized_plan for " << this << std::endl;
    return *optimized_plan_;
  }

  if (!pr_) {
    pr_ = ProfilingRecord::instrumentGraph(prepareGraph(graph, stack));
    auto copy = pr_->graph()->copy();
    // to lower GradOf
    copy->dump();
    std::cout << "getProfilingMode() = " << getProfilingMode() << std::endl;
    //runRequiredPasses(copy);
    LowerGradOf(*copy);
    std::cout << "after LowerGradOf\n";
    copy->dump();
    RemoveExpands(copy);
    CanonicalizeOps(copy);
    EliminateDeadCode(copy);
    profiling_plan_ = ExecutionPlan(copy);
    std::cout << "setting profiling_plan_ for " << this << std::endl;
    copy->dump();
    // fall-through
  }

  if (!pr_->ready()) {
    std::cout << "still profiling for " << this << std::endl;
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
  std::cout << "before running runRequiredPasses\n";
  copy->dump();
  runRequiredPasses(copy);
  if (needsGradient(copy)) {
    GRAPH_DEBUG("needs grad");
    auto diff_nodes = CreateAutodiffSubgraphs(
        copy, getAutodiffSubgraphInlining() ? autodiffSubgraphNodeThreshold : 1);
    for (Node* dnode : diff_nodes) {
      auto diff_graph = std::move(dnode->g(attr::Subgraph));
      Gradient gradient = differentiate(diff_graph);
      std::cout << "gradient.f = \n";
      gradient.f->dump();
      runOptimization(gradient.f);
       // run non diff optimization on the forward graph
      runNondiffOptimization(gradient.f);
      packGradient(gradient, dnode);
    }
    InlineAutodiffSubgraphs(
        copy, getAutodiffSubgraphInlining() ? autodiffSubgraphInlineThreshold : 1);
  }
  GRAPH_DUMP("InlineAutodiffSubgraphs: ", copy);
  ConstantPropagation(copy);
  runOptimization(copy);
  runNondiffOptimization(copy);
  EliminateDeadCode(copy);
  // cache
  optimized_plan_ = ExecutionPlan(copy);
  std::cout << "setting optimized_plan for " << this << std::endl;
  copy->dump();
  GRAPH_DUMP("ExecutionPlan: ", copy);
  return *optimized_plan_;
}


GraphExecutorState ProfilingGraphExecutorImpl::getDebugState() {
  GraphExecutorState state;

  std::cout << "graph in debugstate\n";
  graph->dump();
  if (this->pr_)
  {
    std::cout << "pr_ set\n";
  }
  std::cout << "getDebugState: optimized_plan for " << this << std::endl;
  TORCH_INTERNAL_ASSERT(optimized_plan_);
  auto opt_plan = *optimized_plan_;
  state.execution_plans.emplace(ArgumentSpec{0, 0}, opt_plan);
  return state;
}

} // namespace jit
} // namespace torch
