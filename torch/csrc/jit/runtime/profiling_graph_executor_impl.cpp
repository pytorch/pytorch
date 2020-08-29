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
    if (n->kind() == prim::profile) {
      auto type = n->ty(attr::profiled_type)->expect<TensorType>();
      if (type->requiresGrad() && *type->requiresGrad()) {
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
  GRAPH_DUMP("Before LowerGradOf (beginning of runNooptPassPipeline)", graph);
  LowerGradOf(*graph);
  GRAPH_DUMP("After LowerGradOf, before RemoveExpands", graph);
  RemoveExpands(graph);
  GRAPH_DUMP("After RemoveExpands, before CanonicalizeOps", graph);
  CanonicalizeOps(graph);
  GRAPH_DUMP("After CanonicalizeOps, before EliminateDeadCode", graph);
  EliminateDeadCode(graph);
  GRAPH_DUMP("After EliminateDeadCode (end of runNooptPassPipeline)", graph);
}

void runPreAutodiffPassPipeline(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP(
      "Before InsertGuards (beginning of runPreAutodiffPassPipeline)", graph);

  if (tensorExprFuserEnabled()) {
    // With TE fuser we don't generate bailouts
    LowerGradOf(*graph);
    GRAPH_DUMP("After LowerGradOf, before specializeAutogradZero", graph);
  } else {
    InsertGuards(graph);
    GRAPH_DUMP("After InsertGuards, before LowerGradOf", graph);
    LowerGradOf(*graph);
    GRAPH_DUMP("After LowerGradOf, before EliminateRedundantGuards", graph);
    EliminateRedundantGuards(graph);
    GRAPH_DUMP("After EliminateRedundantGuards, before InsertBailOuts", graph);
    InsertBailOuts(graph);
    GRAPH_DUMP("After InsertBailOuts, before specializeAutogradZero", graph);
  }

  specializeAutogradZero(graph);
  GRAPH_DUMP("After specializeAutogradZero", graph);
  // runRequiredPasses
  {
    RemoveExpands(graph);
    GRAPH_DUMP("After RemoveExpands, before CanonicalizeOps", graph);
    CanonicalizeOps(graph);
    GRAPH_DUMP("After CanonicalizeOps, before EliminateDeadCode", graph);
    EliminateDeadCode(graph);
    GRAPH_DUMP("After EliminateDeadCode", graph);
  }
  PeepholeOptimize(graph);
  GRAPH_DUMP("After PeepholeOptimize, before ConstantPropagation", graph);
  ConstantPropagation(graph);

  // runOptimization:
  {
    EliminateDeadCode(graph);
    GRAPH_DUMP(
        "After EliminateDeadCode, before EliminateCommonSubexpression", graph);
    EliminateCommonSubexpression(graph);
    GRAPH_DUMP(
        "After EliminateCommonSubexpression, before PeepholeOptimize", graph);

    PeepholeOptimize(graph);
    GRAPH_DUMP("After PeepholeOptimize, before ConstantPropagation", graph);
    ConstantPropagation(graph);
    GRAPH_DUMP("After ConstantPropagation, before ConstantPooling", graph);
    ConstantPooling(graph);
    GRAPH_DUMP("After ConstantPooling, before UnrollLoops", graph);

    UnrollLoops(graph);
    GRAPH_DUMP("After UnrollLoops, before RemoveListMutation", graph);
    // run again with unrolled loops
    RemoveListMutation(graph);
    GRAPH_DUMP("After RemoveListMutation, before PeepholeOptimize", graph);
    PeepholeOptimize(graph);
    GRAPH_DUMP("After PeepholeOptimize, before ConstantPropagation", graph);
    ConstantPropagation(graph);
    GRAPH_DUMP(
        "After ConstantPropagation, before EliminateCommonSubexpression",
        graph);

    EliminateCommonSubexpression(graph);
    GRAPH_DUMP(
        "After EliminateCommonSubexpression, before CheckInplace", graph);

    CheckInplace(graph);
  }
  GRAPH_DUMP("After CheckInplace (end of runPreAutodiffPassPipeline)", graph);
}

void runDiffGraphPasses(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP(
      "Before EliminateDeadCode (beginning of runDiffGraphPasses)", graph);
  // runOptimization:
  {
    // Basic graph preprocessing to eliminate noise.
    EliminateDeadCode(graph);
    GRAPH_DUMP(
        "After EliminateDeadCode, before EliminateCommonSubexpression", graph);
    EliminateCommonSubexpression(graph);
    GRAPH_DUMP(
        "After EliminateCommonSubexpression, before PeepholeOptimize", graph);

    PeepholeOptimize(graph);
    GRAPH_DUMP("After PeepholeOptimize, before ConstantPropagation", graph);
    ConstantPropagation(graph);
    GRAPH_DUMP("After ConstantPropagation, before ConstantPooling", graph);
    ConstantPooling(graph);
    GRAPH_DUMP("After ConstantPooling, before UnrollLoops", graph);

    UnrollLoops(graph);
    GRAPH_DUMP("After UnrollLoops, before RemoveListMutation", graph);
    // run again with unrolled loops
    RemoveListMutation(graph);
    GRAPH_DUMP("After RemoveListMutation, before PeepholeOptimize", graph);
    PeepholeOptimize(graph);
    GRAPH_DUMP("After PeepholeOptimize, before ConstantPropagation", graph);
    ConstantPropagation(graph);
    GRAPH_DUMP(
        "After ConstantPropagation, before EliminateCommonSubexpression",
        graph);

    EliminateCommonSubexpression(graph);
    GRAPH_DUMP(
        "After EliminateCommonSubexpression, before CheckInplace", graph);

    CheckInplace(graph);
  }
  GRAPH_DUMP("After CheckInplace, before customPrePasses", graph);

  // runNondiffOptimization
  {
    // Run custom passes that different backends can register.
    for (const auto& passPair : getCustomPrePasses()) {
      passPair.first(graph);
    }
    GRAPH_DUMP("After customPrePasses, before LowerSimpleTuples", graph);

    // TupleConstruct / TupleUnpack pairs can still be present at this point
    // and must be removed for fusion.
    LowerSimpleTuples(graph);
    GRAPH_DUMP("After LowerSimpleTuples, before BatchMM", graph);

    // Rewrite subgraphs with many MMs into expressions that batch them.
    BatchMM(graph);
    GRAPH_DUMP("After BatchMM, before Fusion", graph);

    if (tensorExprFuserEnabled()) {
      FuseTensorExprs(graph);
    } else {
      FuseGraph(graph, true);
    }
    GRAPH_DUMP("After Fusion, before customPostPasses", graph);

    // Run custom post-fusion passes
    for (const auto& passPair : getCustomPostPasses()) {
      passPair.first(graph);
    }
  }
  GRAPH_DUMP("After customPostPasses (end of runDiffGraphPasses)", graph);
}

void runNoGradOptimizations(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP(
      "After customPostPasses (beginning of runNoGradOptimizations)", graph);
  // runNondiffOptimization
  {
    // Run custom passes that different backends can register.
    for (const auto& passPair : getCustomPrePasses()) {
      passPair.first(graph);
    }
    GRAPH_DUMP("After customPrePasses, before LowerSimpleTuples", graph);

    // TupleConstruct / TupleUnpack pairs can still be present at this point
    // and must be removed for fusion.
    LowerSimpleTuples(graph);
    GRAPH_DUMP("After LowerSimpleTuples, before BatchMM", graph);

    // Rewrite subgraphs with many MMs into expressions that batch them.
    BatchMM(graph);
    GRAPH_DUMP("After BatchMM, before Fusion", graph);

    if (tensorExprFuserEnabled()) {
      FuseTensorExprs(graph);
    } else {
      FuseGraph(graph, true);
    }
    GRAPH_DUMP("After Fusion, before customPostPasses", graph);

    // Run custom post-fusion passes
    for (const auto& passPair : getCustomPostPasses()) {
      passPair.first(graph);
    }
  }
  GRAPH_DUMP("After customPostPasses (end of runNoGradOptimizations)", graph);
}

void ProfilingGraphExecutorImpl::runProfilingOptimizations(
    std::shared_ptr<Graph>& copy) {
  GRAPH_DUMP("Before runProfilingOptimizations:", copy);
  if (!getGraphExecutorOptimize()) {
    runNooptPassPipeline(copy);
    return;
  }

  runPreAutodiffPassPipeline(copy);

  if (needsGradientInProfilingMode(copy->block())) {
    auto diff_nodes = CreateAutodiffSubgraphs(
        copy,
        getAutodiffSubgraphInlining() ? autodiffSubgraphNodeThreshold : 1);
    GRAPH_DUMP("After CreateAutodiffSubgraphs", copy);
    size_t idx = 0;
    for (Node* dnode : diff_nodes) {
      GRAPH_DEBUG("Optimizing diff node ", idx);
      auto diff_graph = std::move(dnode->g(attr::Subgraph));
      Gradient gradient = differentiate(diff_graph);
      GRAPH_DUMP("Forward graph:", gradient.f);
      GRAPH_DUMP("Backward graph:", gradient.df);
      runDiffGraphPasses(gradient.f);
      // replaces fallback graphs inserted by TE Fuser
      replaceFallbackGraphWithFallbackFunction(gradient.f->block());
      packGradient(gradient, dnode);
      GRAPH_DEBUG("Finished optimizing diff node ", idx++);
    }
    InlineAutodiffSubgraphs(
        copy,
        getAutodiffSubgraphInlining() ? autodiffSubgraphInlineThreshold : 1);
    GRAPH_DUMP("After InlineAutodiffSubgraphs", copy);
  } else {
    runNoGradOptimizations(copy);
  }
  EliminateDeadCode(copy);
  GRAPH_DUMP("After runProfilingOptimizations:", copy);
}

void ProfilingGraphExecutorImpl::runProfilingInsensitiveOptimizations(
    std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP(
      "Before ClearProfilingInformation (beginning of runProfilingInsensitiveOptimizations)",
      graph);
  ClearProfilingInformation(graph);
  GRAPH_DUMP("After ClearProfilingInformation, before LowerGradOf", graph);
  LowerGradOf(*graph);
  GRAPH_DUMP("After LowerGradOf, before ClearUndefinedness", graph);
  // clear any residual undefinedness
  // as double backward graph inputs'
  // may carry over undefinedness
  // from profiled backward graphs
  ClearUndefinedness(graph);
  // runRequiredPasses
  {
    GRAPH_DUMP("After ClearUndefinedness, before RemoveExpands", graph);
    RemoveExpands(graph);
    GRAPH_DUMP("After RemoveExpands, before CanonicalizeOps", graph);
    CanonicalizeOps(graph);
    GRAPH_DUMP("After CanonicalizeOps, before EliminateDeadCode", graph);
    EliminateDeadCode(graph);
  }
  if (!getGraphExecutorOptimize()) {
    GRAPH_DUMP(
        "After EliminateDeadCode (end of runProfilingInsensitiveOptimizations)",
        graph);
    return;
  }

  GRAPH_DUMP("After EliminateDeadCode, before DecomposeOps", graph);
  DecomposeOps(graph);
  GRAPH_DUMP("After DecomposeOps, before ConstantPropagation", graph);
  ConstantPropagation(graph);
  GRAPH_DUMP("After ConstantPropagation, before EliminateDeadCode", graph);
  EliminateDeadCode(graph);
  GRAPH_DUMP(
      "After EliminateDeadCode, before EliminateCommonSubexpression", graph);
  EliminateCommonSubexpression(graph);
  GRAPH_DUMP(
      "After EliminateCommonSubexpression, before ConstantPooling", graph);
  ConstantPooling(graph);
  GRAPH_DUMP("After ConstantPooling, before PeepholeOptimize", graph);
  PeepholeOptimize(graph);
  GRAPH_DUMP("After PeepholeOptimize, before EliminateDeadCode", graph);
  EliminateDeadCode(graph);
  GRAPH_DUMP("After EliminateDeadCode, before LowerSimpleTuples", graph);
  LowerSimpleTuples(graph);
  GRAPH_DUMP("After LowerSimpleTuples, before CheckInplace", graph);
  CheckInplace(graph);
  GRAPH_DUMP(
      "After CheckInplace (end of runProfilingInsensitiveOptimizations)",
      graph);
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

  // if tensorExprFuserEnabled() returns true we need to persist the very first
  // time ProfilingGraphExecutorImpl is called, so we can update it correctly
  // for fallback functions in ProfilingGraphExecutorImpl Else,
  // getPlanFor(remaining_bailout_depth) is corrected and persisted by the Code
  // object in interpreter.
  if (!remaining_bailout_depth_.has_value() || !tensorExprFuserEnabled()) {
    remaining_bailout_depth_ = remaining_bailout_depth;
  }

  if (optimized_plan_) {
    return *optimized_plan_;
  }

  // simple executor
  if (*remaining_bailout_depth_ == 0) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy);
    GRAPH_DUMP("Optimized SimpleExecutor Graph : ", copy);
    optimized_plan_ = ExecutionPlan(copy, function_name_);
    return *optimized_plan_;
  }

  // if a profiling graph hasn't been created yet
  if (!pr_) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy);
    pr_ = ProfilingRecord::instrumentGraph(copy);
    GRAPH_DUMP("Profiled Graph: ", pr_->graph());
    profiling_plan_ = ExecutionPlan(pr_->graph(), function_name_);
    // fall-through
  }

  // profile until a graph is ready
  if (!pr_->ready()) {
    return *profiling_plan_;
  }

  auto copy = pr_->graph()->copy();
  ProfilingRecord::removeProfileCounter(copy->block());
  runProfilingOptimizations(copy);
  // replaces a fallback graph inserted by
  // specialize_autogradzero if one exists
  replaceFallbackGraphWithFallbackFunction(copy->block());
  GRAPH_DUMP("Optimized Graph: ", copy);
  // cache
  optimized_plan_ =
      ExecutionPlan(copy, function_name_, *remaining_bailout_depth_);
  return *optimized_plan_;
}

GraphExecutorState ProfilingGraphExecutorImpl::getDebugState() {
  GraphExecutorState state;
  TORCH_INTERNAL_ASSERT(optimized_plan_);
  auto opt_plan = *optimized_plan_;
  state.execution_plans.emplace(ArgumentSpec{0, 0}, opt_plan);
  return state;
}

void replaceBlockWithFallbackGraph(Block* b) {
  auto graph = std::make_shared<Graph>();
  auto value_map = [](Value* v) { return v; };
  graph->block()->cloneFrom(b, value_map);
  auto fallback = b->owningGraph()->create(
      prim::FallbackGraph, b->inputs(), b->outputs().size());
  fallback->g_(attr::Subgraph, graph);
  b->prependNode(fallback);

  for (size_t i = 0; i < b->outputs().size(); i++) {
    fallback->output(i)->setType(b->outputs()[i]->type());
    fallback->output(i)->copyMetadata(b->outputs()[i]);
    b->replaceOutput(i, fallback->output(i));
  }

  for (auto it = b->nodes().rbegin(); it != fallback->iterator(); it++) {
    it.destroyCurrent();
  }
}

static Function* createFallbackPathFunction(
    Block* b,
    const std::string& function_name) {
  auto value_map = [](Value* v) { return v; };
  auto graph = std::make_shared<Graph>();
  graph->block()->cloneFrom(b, value_map);

  auto otypes = c10::fmap(
      graph->return_node()->inputs(), [](Value* v) { return v->type(); });
  // a GraphFunction call only have one output, so all the outputs
  // need to be packed into a tuple
  auto tuple_type = TupleType::create(otypes);
  auto return_tuple = graph->createTuple(graph->return_node()->inputs());
  graph->appendNode(return_tuple);
  for (int i = static_cast<int>(graph->outputs().size()) - 1; i >= 0; i--) {
    graph->eraseOutput(i);
  }
  graph->registerOutput(return_tuple->output());
  return new GraphFunction(function_name, graph, nullptr);
}

Node* insertFallbackFunctionCall(
    Graph* graph,
    Function* func,
    ArrayRef<Value*> inputs) {
  auto tuple_type = func->graph()->return_node()->input(0)->type();
  Value* fn_constant = graph->insertNode(graph->create(prim::Constant))
                           ->s_(attr::name, func->name())
                           ->i_(Symbol::attr("fallback"), 1)
                           ->output()
                           ->setType(FunctionType::create(func));
  std::vector<Value*> func_call_inputs = {fn_constant};
  func_call_inputs.insert(func_call_inputs.end(), inputs.begin(), inputs.end());
  Value* result =
      graph->insertNode(graph->create(prim::CallFunction, func_call_inputs))
          ->output()
          ->setType(tuple_type);

  auto fun_unpack_tuple = graph->insertNode(graph->createTupleUnpack(result));
  return fun_unpack_tuple;
}

void ProfilingGraphExecutorImpl::replaceFallbackGraphWithFallbackFunction(
    Block* b) {
  Stack s;
  for (auto it = b->nodes().begin(); it != b->nodes().end();) {
    if (it->kind() == prim::FallbackGraph) {
      auto fallback_func = createFallbackPathFunction(
          it->g(attr::Subgraph)->block(), "fallback_function");
      TORCH_INTERNAL_ASSERT(*remaining_bailout_depth_ > 0);
      GRAPH_DEBUG(
          "getPlanFor for", getHeader(*it), " ", *remaining_bailout_depth_);
      fallback_func->get_executor().getPlanFor(
          s, *remaining_bailout_depth_ - 1);
      fallback_functions_.emplace_back(fallback_func);
      WithInsertPoint wip{*it};
      auto function_call = insertFallbackFunctionCall(
          b->owningGraph(), fallback_func, it->inputs());
      for (size_t i = 0; i < function_call->outputs().size(); i++) {
        it->output(i)->replaceAllUsesWith(function_call->output(i));
      }
      it.destroyCurrent();
    } else {
      for (Block* ib : it->blocks()) {
        replaceFallbackGraphWithFallbackFunction(ib);
      }
      it++;
    }
  }
}

} // namespace jit
} // namespace torch
