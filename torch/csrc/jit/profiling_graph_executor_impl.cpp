#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/canonicalize_ops.h>
#include <torch/csrc/jit/passes/clear_undefinedness.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
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

#if defined (FBCODE_CAFFE2) || defined (C10_MOBILE)
static std::atomic<bool> executor_mode{false};
static std::atomic<bool> profiling_mode{false};
#else
static std::atomic<bool> executor_mode{true};
static std::atomic<bool> profiling_mode{true};
#endif


std::atomic<bool>& getProfilingMode() {
  return profiling_mode;
}
std::atomic<bool>& getExecutorMode() {
  return executor_mode;
}

static bool needsGradientInProfilingMode(Block* b) {
  for (auto n : b->nodes()) {
    if (n->kind() == prim::BailOut) {
      auto ptt = n->output()->type()->expect<TensorType>();
      if (ptt->requiresGrad() && *ptt->requiresGrad()) {
        return true;
      }
    }

    for (auto ib : n->blocks()) {
      if (needsGradientInProfilingMode(ib)) {
        return true;
      }
    }
  }
  return false;
}

std::shared_ptr<Graph> ProfilingGraphExecutorImpl::prepareGraph(
    const std::shared_ptr<Graph>& graph,
    Stack& stack) {
  auto g = graph->copy();
  return g;
}

ProfilingGraphExecutorImpl::ProfilingGraphExecutorImpl(
    const std::shared_ptr<Graph>& graph)
    : GraphExecutorImplBase(graph) {}

ExecutionPlan ProfilingGraphExecutorImpl::getPlanFor(Stack& stack) {
  std::lock_guard<std::mutex> lock(compile_mutex);
  GRAPH_DEBUG("Running ProfilingGraphExecutorImpl ", this);
  if (optimized_plan_) {
    return *optimized_plan_;
  }

  std::shared_ptr<Graph> copy;
  if (getProfilingMode()) {
    if (!pr_) {
      pr_ = ProfilingRecord::instrumentGraph(prepareGraph(graph, stack));
      auto copy = pr_->graph()->copy();
      LowerGradOf(*copy);
      specializeAutogradZero(*copy);
      runRequiredPasses(copy);
      GRAPH_DUMP("Profiled Graph: ", copy);
      profiling_plan_ = ExecutionPlan(copy);
      // fall-through
    }

    if (!pr_->ready()) {
      return *profiling_plan_;
    }
    copy = pr_->graph()->copy();

  } else {
    copy = graph->copy();
  }

  if (!getGraphExecutorOptimize()) {
    runRequiredPasses(copy);
    optimized_plan_ = ExecutionPlan(copy);
    return *optimized_plan_;
  }

  InsertGuards(copy);
  LowerGradOf(*copy);
  if (getProfilingMode()) {
    EliminateRedundantGuards(copy);
    InsertBailOuts(copy);
    GRAPH_DUMP("After InsertBailOuts: ", copy);
  }

  specializeAutogradZero(*copy);
  if (!getProfilingMode()) {
    ClearUndefinedness(copy);
  }

  runRequiredPasses(copy);
  ConstantPropagation(copy);
  runOptimization(copy);

  // TODO: insert grad propagation
  bool needs_gradient = getProfilingMode()
      ? needsGradientInProfilingMode(copy->block())
      : true;
  if (needs_gradient) {
    // for Simple Executor skip creating autodiff graphs
    // and let autograd handle backward for us
    if (getProfilingMode()) {
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
  } else {
    runNondiffOptimization(copy);
  }
  EliminateDeadCode(copy);
  GRAPH_DUMP("Optimized Graph : ", copy);
  // cache
  optimized_plan_ = ExecutionPlan(copy);
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
