#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
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
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

C10_DECLARE_bool();

C10_DEFINE_bool(
    torch_jit_enable_new_executor,
    true,
    "If this flag is set to false TorchScript will be using the legacy/original executor");

namespace torch {
namespace jit {

// TODO: keep the else clause for trial runs
#if defined(FBCODE_CAFFE2) || defined(C10_MOBILE)
static std::atomic<bool> executor_mode{true};
static std::atomic<bool> profiling_mode{false};
#else
static std::atomic<bool> executor_mode{true};
static std::atomic<bool> profiling_mode{false};
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

void runNooptPassPipeline(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG(
      "Before LowerGradOf (beginning of runNooptPassPipeline)\n", *graph);
  LowerGradOf(*graph);
  GRAPH_DEBUG("After LowerGradOf, before RemoveExpands\n", *graph);
  RemoveExpands(graph);
  GRAPH_DEBUG("After RemoveExpands, before CanonicalizeOps\n", *graph);
  CanonicalizeOps(graph);
  GRAPH_DEBUG("After CanonicalizeOps, before EliminateDeadCode\n", *graph);
  EliminateDeadCode(graph);
  GRAPH_DEBUG(
      "After EliminateDeadCode (end of runNooptPassPipeline)\n", *graph);
}

void runPreAutodiffPassPipeline(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG(
      "Before InsertGuards (beginning of runPreAutodiffPassPipeline)\n",
      *graph);

  if (tensorExprFuserEnabled()) {
    // With TE fuser we don't generate bailouts
    LowerGradOf(*graph);
    GRAPH_DEBUG("After LowerGradOf, before specializeAutogradZero\n", *graph);
  } else {
    InsertGuards(graph);
    GRAPH_DEBUG("After InsertGuards, before LowerGradOf\n", *graph);
    LowerGradOf(*graph);
    GRAPH_DEBUG("After LowerGradOf, before EliminateRedundantGuards\n", *graph);
    EliminateRedundantGuards(graph);
    GRAPH_DEBUG(
        "After EliminateRedundantGuards, before InsertBailOuts\n", *graph);
    InsertBailOuts(graph);
    GRAPH_DEBUG(
        "After InsertBailOuts, before specializeAutogradZero\n", *graph);
  }

  specializeAutogradZero(*graph);
  GRAPH_DEBUG("After specializeAutogradZero\n", *graph);
  // runRequiredPasses
  {
    RemoveExpands(graph);
    GRAPH_DEBUG("After RemoveExpands, before CanonicalizeOps\n", *graph);
    CanonicalizeOps(graph);
    GRAPH_DEBUG("After CanonicalizeOps, before EliminateDeadCode\n", *graph);
    EliminateDeadCode(graph);
    GRAPH_DEBUG("After EliminateDeadCode", *graph);
  }
  PeepholeOptimize(graph);
  GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
  ConstantPropagation(graph);

  // runOptimization:
  {
    EliminateDeadCode(graph);
    GRAPH_DEBUG(
        "After EliminateDeadCode, before EliminateCommonSubexpression\n",
        *graph);
    EliminateCommonSubexpression(graph);
    GRAPH_DEBUG(
        "After EliminateCommonSubexpression, before PeepholeOptimize\n",
        *graph);

    PeepholeOptimize(graph);
    GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
    ConstantPropagation(graph);
    GRAPH_DEBUG("After ConstantPropagation, before ConstantPooling\n", *graph);
    ConstantPooling(graph);
    GRAPH_DEBUG("After ConstantPooling, before UnrollLoops\n", *graph);

    UnrollLoops(graph);
    GRAPH_DEBUG("After UnrollLoops, before RemoveListMutation\n", *graph);
    // run again with unrolled loops
    RemoveListMutation(graph);
    GRAPH_DEBUG("After RemoveListMutation, before PeepholeOptimize\n", *graph);
    PeepholeOptimize(graph);
    GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
    ConstantPropagation(graph);
    GRAPH_DEBUG(
        "After ConstantPropagation, before EliminateCommonSubexpression\n",
        *graph);

    EliminateCommonSubexpression(graph);
    GRAPH_DEBUG(
        "After EliminateCommonSubexpression, before CheckInplace\n", *graph);

    CheckInplace(graph);
  }
  GRAPH_DEBUG(
      "After CheckInplace (end of runPreAutodiffPassPipeline)\n", *graph);
}

void runDiffGraphPasses(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG(
      "Before EliminateDeadCode (beginning of runDiffGraphPasses)\n", *graph);
  // runOptimization:
  {
    // Basic graph preprocessing to eliminate noise.
    EliminateDeadCode(graph);
    GRAPH_DEBUG(
        "After EliminateDeadCode, before EliminateCommonSubexpression\n",
        *graph);
    EliminateCommonSubexpression(graph);
    GRAPH_DEBUG(
        "After EliminateCommonSubexpression, before PeepholeOptimize\n",
        *graph);

    PeepholeOptimize(graph);
    GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
    ConstantPropagation(graph);
    GRAPH_DEBUG("After ConstantPropagation, before ConstantPooling\n", *graph);
    ConstantPooling(graph);
    GRAPH_DEBUG("After ConstantPooling, before UnrollLoops\n", *graph);

    UnrollLoops(graph);
    GRAPH_DEBUG("After UnrollLoops, before RemoveListMutation\n", *graph);
    // run again with unrolled loops
    RemoveListMutation(graph);
    GRAPH_DEBUG("After RemoveListMutation, before PeepholeOptimize\n", *graph);
    PeepholeOptimize(graph);
    GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
    ConstantPropagation(graph);
    GRAPH_DEBUG(
        "After ConstantPropagation, before EliminateCommonSubexpression\n",
        *graph);

    EliminateCommonSubexpression(graph);
    GRAPH_DEBUG(
        "After EliminateCommonSubexpression, before CheckInplace\n", *graph);

    CheckInplace(graph);
  }
  GRAPH_DEBUG("After CheckInplace, before customPrePasses\n", *graph);

  // runNondiffOptimization
  {
    // Run custom passes that different backends can register.
    for (const auto& passPair : getCustomPrePasses()) {
      passPair.first(graph);
    }
    GRAPH_DEBUG("After customPrePasses, before LowerSimpleTuples\n", *graph);

    // TupleConstruct / TupleUnpack pairs can still be present at this point
    // and must be removed for fusion.
    LowerSimpleTuples(graph);
    GRAPH_DEBUG("After LowerSimpleTuples, before BatchMM\n", *graph);

    // Rewrite subgraphs with many MMs into expressions that batch them.
    BatchMM(graph);
    GRAPH_DEBUG("After BatchMM, before Fusion\n", *graph);

    if (tensorExprFuserEnabled()) {
      FuseTensorExprs(graph);
    } else {
      FuseGraph(graph, true);
    }
    GRAPH_DEBUG("After Fusion, before customPostPasses\n", *graph);

    // Run custom post-fusion passes
    for (const auto& passPair : getCustomPostPasses()) {
      passPair.first(graph);
    }
  }
  GRAPH_DEBUG("After customPostPasses (end of runDiffGraphPasses)\n", *graph);
}

void runNoGradOptimizations(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG(
      "After customPostPasses (beginning of runNoGradOptimizations)\n", *graph);
  // runNondiffOptimization
  {
    // Run custom passes that different backends can register.
    for (const auto& passPair : getCustomPrePasses()) {
      passPair.first(graph);
    }
    GRAPH_DEBUG("After customPrePasses, before LowerSimpleTuples\n", *graph);

    // TupleConstruct / TupleUnpack pairs can still be present at this point
    // and must be removed for fusion.
    LowerSimpleTuples(graph);
    GRAPH_DEBUG("After LowerSimpleTuples, before BatchMM\n", *graph);

    // Rewrite subgraphs with many MMs into expressions that batch them.
    BatchMM(graph);
    GRAPH_DEBUG("After BatchMM, before Fusion\n", *graph);

    if (tensorExprFuserEnabled()) {
      FuseTensorExprs(graph);
    } else {
      FuseGraph(graph, true);
    }
    GRAPH_DEBUG("After Fusion, before customPostPasses\n", *graph);

    // Run custom post-fusion passes
    for (const auto& passPair : getCustomPostPasses()) {
      passPair.first(graph);
    }
  }
  GRAPH_DEBUG(
      "After customPostPasses (end of runNoGradOptimizations)\n", *graph);
}

void ProfilingGraphExecutorImpl::runProfilingOptimizations(
    std::shared_ptr<Graph>& copy) {
  GRAPH_DEBUG("Before runProfilingOptimizations:\n", *copy);
  if (!getGraphExecutorOptimize()) {
    runNooptPassPipeline(copy);
    return;
  }

  runPreAutodiffPassPipeline(copy);

  if (needsGradientInProfilingMode(copy->block())) {
    auto diff_nodes = CreateAutodiffSubgraphs(
        copy,
        getAutodiffSubgraphInlining() ? autodiffSubgraphNodeThreshold : 1);
    GRAPH_DEBUG("After CreateAutodiffSubgraphs\n", *copy);
    size_t idx = 0;
    for (Node* dnode : diff_nodes) {
      GRAPH_DEBUG("Optimizing diff node ", idx);
      auto diff_graph = std::move(dnode->g(attr::Subgraph));
      Gradient gradient = differentiate(diff_graph);
      GRAPH_DEBUG("Forward graph:\n", *(gradient.f));
      GRAPH_DEBUG("Backward graph:\n", *(gradient.df));
      runDiffGraphPasses(gradient.f);
      packGradient(gradient, dnode);
      GRAPH_DEBUG("Finished optimizing diff node ", idx++);
    }
    InlineAutodiffSubgraphs(
        copy,
        getAutodiffSubgraphInlining() ? autodiffSubgraphInlineThreshold : 1);
    GRAPH_DEBUG("After InlineAutodiffSubgraphs\n", *copy);
  } else {
    runNoGradOptimizations(copy);
  }
  EliminateDeadCode(copy);
  GRAPH_DEBUG("After runProfilingOptimizations:\n", *copy);
}

void ProfilingGraphExecutorImpl::runProfilingInsensitiveOptimizations(
    std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG(
      "Before ClearProfilingInformation (beginning of runProfilingInsensitiveOptimizations)\n",
      *graph);
  ClearProfilingInformation(graph);
  GRAPH_DEBUG("After ClearProfilingInformation, before LowerGradOf\n", *graph);
  LowerGradOf(*graph);
  GRAPH_DEBUG("After LowerGradOf, before ClearUndefinedness\n", *graph);
  // clear any residual undefinedness
  // as double backward graph inputs'
  // may carry over undefinedness
  // from profiled backward graphs
  ClearUndefinedness(graph);
  // runRequiredPasses
  {
    GRAPH_DEBUG("After ClearUndefinedness, before RemoveExpands\n", *graph);
    RemoveExpands(graph);
    GRAPH_DEBUG("After RemoveExpands, before CanonicalizeOps\n", *graph);
    CanonicalizeOps(graph);
    GRAPH_DEBUG("After CanonicalizeOps, before EliminateDeadCode\n", *graph);
    EliminateDeadCode(graph);
  }
  if (!getGraphExecutorOptimize()) {
    GRAPH_DEBUG(
        "After EliminateDeadCode (end of runProfilingInsensitiveOptimizations)\n",
        *graph);
    return;
  }

  GRAPH_DEBUG("After EliminateDeadCode, before DecomposeOps\n", *graph);
  DecomposeOps(graph);
  GRAPH_DEBUG("After DecomposeOps, before ConstantPropagation\n", *graph);
  ConstantPropagation(graph);
  GRAPH_DEBUG("After ConstantPropagation, before EliminateDeadCode\n", *graph);
  EliminateDeadCode(graph);
  GRAPH_DEBUG(
      "After EliminateDeadCode, before EliminateCommonSubexpression\n", *graph);
  EliminateCommonSubexpression(graph);
  GRAPH_DEBUG(
      "After EliminateCommonSubexpression, before ConstantPooling\n", *graph);
  ConstantPooling(graph);
  GRAPH_DEBUG("After ConstantPooling, before PeepholeOptimize\n", *graph);
  PeepholeOptimize(graph);
  GRAPH_DEBUG("After PeepholeOptimize, before EliminateDeadCode\n", *graph);
  EliminateDeadCode(graph);
  GRAPH_DEBUG("After EliminateDeadCode, before LowerSimpleTuples\n", *graph);
  LowerSimpleTuples(graph);
  GRAPH_DEBUG("After LowerSimpleTuples, before CheckInplace\n", *graph);
  CheckInplace(graph);
  GRAPH_DEBUG(
      "After CheckInplace (end of runProfilingInsensitiveOptimizations)\n",
      *graph);
}

ProfilingGraphExecutorImpl::ProfilingGraphExecutorImpl(
    const std::shared_ptr<Graph>& graph,
    std::string function_name)
    : GraphExecutorImplBase(graph, std::move(function_name)) {}

ExecutionPlan ProfilingGraphExecutorImpl::getPlanFor(
    Stack& stack,
    size_t remaining_bailout_depth) {
  std::lock_guard<std::mutex> lock(compile_mutex);
  GRAPH_DEBUG("Running ProfilingGraphExecutorImpl ", this);

  if (optimized_plan_) {
    GRAPH_DUMP("plan already optimized:", graph);
    return *optimized_plan_;
  }

  // simple executor
  if (remaining_bailout_depth == 0) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy);
    GRAPH_DUMP("Optimized SimpleExecutor Graph: ", copy);
    optimized_plan_ = ExecutionPlan(copy, function_name_);
    return *optimized_plan_;
  }

  // if a profiling graph hasn't been created yet
  if (!pr_) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy);
    if (remaining_bailout_depth == getBailoutDepth()) {
      PeelProfilingLoops(copy);
    }
    pr_ = ProfilingRecord::instrumentGraph(copy);
    auto pr_copy = pr_->graph()->copy();
    GRAPH_DUMP("Profiled Graph: ", pr_copy);
    profiling_plan_ = ExecutionPlan(pr_copy, function_name_);
    // fall-through
  }

  // profile until a graph is ready
  if (!pr_->ready()) {
    return *profiling_plan_;
  }

  auto copy = pr_->graph()->copy();
  ProfilingRecord::removeProfileCounter(copy->block());
  runProfilingOptimizations(copy);
  // cache
  GRAPH_DUMP("Optimized Graph: ", copy);
  optimized_plan_ =
      ExecutionPlan(copy, function_name_, remaining_bailout_depth);
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
