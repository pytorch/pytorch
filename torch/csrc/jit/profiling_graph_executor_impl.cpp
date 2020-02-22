#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/canonicalize_ops.h>
#include <torch/csrc/jit/passes/clear_undefinedness.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/guard_elimination.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/inplace_check.h>
#include <torch/csrc/jit/passes/insert_guards.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/peephole.h>
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

static std::atomic<size_t> num_profiled_runs{1};
static std::atomic<size_t> bailout_depth{1};

std::atomic<bool>& getProfilingMode() {
  return profiling_mode;
}
std::atomic<bool>& getExecutorMode() {
  return executor_mode;
}

std::atomic<size_t>& getNumProfiledRuns() {
  return num_profiled_runs;
}

std::atomic<size_t>& getBailoutDepth() {
  return bailout_depth;
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

void ProfilingGraphExecutorImpl::runProfilingOptimizations(
    std::shared_ptr<Graph>& copy) {
  if (!getGraphExecutorOptimize()) {
    LowerGradOf(*copy);
    runRequiredPasses(copy);
    return;
  }

  InsertGuards(copy);
  LowerGradOf(*copy);
  EliminateRedundantGuards(copy);
  InsertBailOuts(copy);
  GRAPH_DUMP("After InsertBailOuts: ", copy);
  specializeAutogradZero(*copy);

  runRequiredPasses(copy);
  ConstantPropagation(copy);
  runOptimization(copy);

  if (needsGradientInProfilingMode(copy->block())) {
    auto diff_nodes = CreateAutodiffSubgraphs(
        copy,
        getAutodiffSubgraphInlining() ? autodiffSubgraphNodeThreshold : 1);
    for (Node* dnode : diff_nodes) {
      auto diff_graph = std::move(dnode->g(attr::Subgraph));
      Gradient gradient = differentiate(diff_graph);
      runOptimization(gradient.f);
      // run non diff optimization on the forward graph
      runNondiffOptimization(gradient.f);
      packGradient(gradient, dnode);
    }
    InlineAutodiffSubgraphs(
        copy,
        getAutodiffSubgraphInlining() ? autodiffSubgraphInlineThreshold : 1);

  } else {
    runNondiffOptimization(copy);
  }
  EliminateDeadCode(copy);
  GRAPH_DUMP("Optimized Graph : ", copy);
}

void ProfilingGraphExecutorImpl::runProfilingInsensitiveOptimizations(
    std::shared_ptr<Graph>& copy) {
  LowerGradOf(*copy);
  GRAPH_DUMP("runProfilingInsensitiveOptimizations", copy);
  // clear any residual undefinedness
  // as double backward graph inputs'
  // may carry over undefinedness
  // from profiled backward graphs
  ClearUndefinedness(copy);
  runRequiredPasses(copy);
  if (!getGraphExecutorOptimize()) {
    return;
  }

  DecomposeOps(copy);
  ConstantPropagation(copy);
  EliminateDeadCode(copy);
  EliminateCommonSubexpression(copy);
  ConstantPooling(copy);
  PeepholeOptimize(copy);
  EliminateDeadCode(copy);
  CheckInplace(copy);
}

ProfilingGraphExecutorImpl::ProfilingGraphExecutorImpl(
    const std::shared_ptr<Graph>& graph)
    : GraphExecutorImplBase(graph) {}

ExecutionPlan ProfilingGraphExecutorImpl::getPlanFor(
    Stack& stack,
    size_t remaining_bailout_depth) {
  std::lock_guard<std::mutex> lock(compile_mutex);
  GRAPH_DEBUG("Running ProfilingGraphExecutorImpl ", this);

  if (optimized_plan_) {
    return *optimized_plan_;
  }

  // simple executor
  if (remaining_bailout_depth == 0) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy);
    GRAPH_DUMP("Optimized SimpleExecutor Graph : ", copy);
    optimized_plan_ = ExecutionPlan(copy);
    return *optimized_plan_;
  }

  // if a profiling graph hasn't been created yet
  if (!pr_) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy);
    pr_ = ProfilingRecord::instrumentGraph(copy);
    auto pr_copy = pr_->graph()->copy();
    GRAPH_DUMP("Profiled Graph: ", pr_copy);
    profiling_plan_ = ExecutionPlan(pr_copy);
    // fall-through
  }

  // profile until a graph is ready
  if (!pr_->ready()) {
    const static auto merge = std::getenv("PYTORCH_MERGE");
    if (merge) {
      GRAPH_DUMP("Profiled Graph (merge): ", pr_->graph());
    }
    return *profiling_plan_;
  }

  auto copy = pr_->graph()->copy();
  pr_->convertToStaticShapes(copy->block());
  runProfilingOptimizations(copy);
  // cache
  optimized_plan_ = ExecutionPlan(copy, remaining_bailout_depth);
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
