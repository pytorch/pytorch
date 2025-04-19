#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/add_if_then_else.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>
#include <torch/csrc/jit/passes/check_strict_fusion.h>
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
#include <torch/csrc/jit/passes/inliner.h>
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
#include <torch/csrc/jit/passes/update_differentiable_graph_requires_grad.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <chrono>
#include <mutex>
#include <optional>

// clang-format off
C10_DEFINE_bool(
    torch_jit_enable_new_executor,
    true,
    "If this flag is set to false TorchScript will be using the legacy/original executor")

C10_DEFINE_bool(
    torch_jit_disable_warning_prints,
    false,
    "Disables warning.warn prints in TorchScript graph")

C10_DEFINE_bool(
    torch_jit_static_then_dynamic,
    false,
    "fuse on two static compilations then 10 dynamic")

C10_DEFINE_bool(
    torch_jit_always_dynamic,
    false,
    "fuse on 12 dynamic compilations")

C10_DEFINE_bool(
    torch_jit_release_profiling_graph_after_optimization,
    false,
    "After getOptimizedPlanFor release the optimization record for reduction of memory in inference. This is aggressive memory saving, and please be cautious!")

C10_DEFINE_int32(
    torch_jit_release_profiling_graph_delay_in_seconds,
    60,
    "How long to wait before releasing the profiling graph after optimizaiton is done. Only used if torch_jit_release_profiling_graph_after_optimization is set to true.")

constexpr size_t kDefaultNumProfiledRuns = 1;
constexpr size_t kDefaultBailoutDepth = 20;

C10_DEFINE_int64(
    torch_jit_num_profiled_runs,
    kDefaultNumProfiledRuns,
    "Number of profiling runs")
C10_DEFINE_int64(
    torch_jit_bailout_depth,
    kDefaultBailoutDepth,
    "Number of re-specializations")

namespace torch::jit {

namespace {
int32_t getNowInSecs() {
  auto currentTimePoint = std::chrono::system_clock::now();
  auto durationSinceEpoch = std::chrono::duration_cast<std::chrono::seconds>(
      currentTimePoint.time_since_epoch());
  return static_cast<int32_t>(durationSinceEpoch.count());
}
} // namespace

#if defined(C10_MOBILE)
static std::atomic<bool> executor_mode{true};
static std::atomic<bool> profiling_mode{false};
#else
static std::atomic<bool> executor_mode{true};
static std::atomic<bool> profiling_mode{true};
#endif

static std::mutex fusion_strategy_lock;

static FusionStrategy getInitialStrategy() {
  if (FLAGS_torch_jit_always_dynamic) {
    return {{FusionBehavior::DYNAMIC, 12}};
  }
  FusionStrategy mixed = {
      {FusionBehavior::STATIC, 2}, {FusionBehavior::DYNAMIC, 10}};
  if (FLAGS_torch_jit_static_then_dynamic) {
    return mixed;
  }
// TODO remove ifdef
#ifdef FBCODE_CAFFE2
  return {{FusionBehavior::STATIC, 20}};
#endif
  return mixed;
}

// defer initial value so that we can load in gflags
static std::optional<FusionStrategy> fusion_strategy = std::nullopt;

FusionStrategy getFusionStrategy() {
  std::lock_guard<std::mutex> guard(fusion_strategy_lock);
  if (fusion_strategy == std::nullopt) {
    fusion_strategy = getInitialStrategy();
  }
  return *fusion_strategy;
}

FusionStrategy setFusionStrategy(FusionStrategy& strategy) {
  std::lock_guard<std::mutex> guard(fusion_strategy_lock);
  if (fusion_strategy == std::nullopt) {
    fusion_strategy = getInitialStrategy();
  }
  FusionStrategy old_strategy = *fusion_strategy;
  fusion_strategy = strategy;
  return old_strategy;
}

static std::atomic<size_t> num_profiled_runs{kDefaultNumProfiledRuns};

std::atomic<bool>& getProfilingMode() {
  return profiling_mode;
}

std::atomic<bool>& getExecutorMode() {
  return executor_mode;
}

std::atomic<size_t>& getNumProfiledRuns() {
  // Initialize num_profiled_runs from command-line flag.
  static const size_t init = []() {
    return num_profiled_runs = FLAGS_torch_jit_num_profiled_runs;
  }();
  (void)init; // Silence clang-tidy.
  return num_profiled_runs;
}

size_t getBailoutDepth() {
  // Initialize bailout_depth from command-line flag.
  size_t depth = 0;
  for (const auto& pair : getFusionStrategy()) {
    depth += pair.second;
  }
  return depth;
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

// `prim::RequiresGradCheck` guarantees that requires_grad properties
// of input tensors will match the profiled, otherwise a fallback path
// will be triggered. This allow us to prune off gradients in backward
// graph for inputs that don't need gradients. We transfer requires_grad
// properties from inputs to the `prim::DifferentiableGraph` onto inputs to the
// differentiable graph. Autodiff will inspect these properties and prune
// off gradients that aren't required
// `requires_grad` properties from `dnode->outputs()` will also be transferred
[[maybe_unused]] static void setRequiresGradOnDiffGraph(Node* dnode) {
  auto gi = dnode->g(attr::Subgraph)->inputs();
  for (size_t i = 0; i < dnode->inputs().size(); i++) {
    if (auto ty = dnode->input(i)->type()->cast<TensorType>()) {
      auto gi_ty = gi[i]->type()->expect<TensorType>();
      gi[i]->setType(gi_ty->withRequiresGrad(ty->requires_grad()));
      GRAPH_DEBUG(
          "Setting ",
          *gi_ty->withRequiresGrad(ty->requires_grad()),
          " on ",
          gi[i],
          " ",
          gi[i]->debugName());
    }
  }

  // We also need to put requires_grad on outputs within subgraph, so autodiff
  // can  set df_input_vjps and DifferentiableGraphOp can set `requires_grad=`
  // properly
  auto go = dnode->g(attr::Subgraph)->outputs();
  auto set_requires_grad = [](const TensorTypePtr& t, Value* val) -> bool {
    if (t && t->requiresGrad().has_value()) {
      GRAPH_DEBUG("setting type ", *t);
      val->setType(t);
      return true;
    }
    return false;
  };

  for (const auto i : c10::irange(go.size())) {
    auto ty = go[i]->type()->cast<TensorType>();
    if (ty) {
      auto n = go[i]->node();
      auto dno = dnode->outputs().at(i);
      for (auto dno_use : dno->uses()) {
        GRAPH_DEBUG("found user of ", i, " as ", *dno_use.user);
        if (n->kind() == prim::profile) {
          if (set_requires_grad(
                  n->ty(attr::profiled_type)->expect<TensorType>(), go[i])) {
            break;
          }
        } else if (dno_use.user->kind() == prim::profile) {
          if (set_requires_grad(
                  dno_use.user->ty(attr::profiled_type)->expect<TensorType>(),
                  go[i])) {
            break;
          }
        } else if (dno_use.user->kind() == prim::DifferentiableGraph) {
          Value* o =
              dno_use.user->g(attr::Subgraph)->inputs().at(dno_use.offset);
          // Is it safe to not check other uses, because we are inside a
          // DifferentiableGraph?
          auto nn = o->uses().at(0).user;
          if (nn->kind() == prim::profile) {
            if (set_requires_grad(
                    nn->ty(attr::profiled_type)->expect<TensorType>(), go[i])) {
              break;
            }
          }
        }
      }
    }
  }
}

static bool guardDifferentiableGraph(Node* dnode) {
  auto gi = dnode->g(attr::Subgraph)->inputs();
  bool all_inputs_seen = true;
  for (const auto i : c10::irange(gi.size())) {
    auto ty = gi[i]->type()->cast<TensorType>();
    if (ty) {
      auto n = gi[i]->uses().at(0).user;
      auto dni = dnode->inputs().at(i);
      GRAPH_DEBUG("found first user of ", i, " as ", *n);
      if (n->kind() == prim::profile) {
        GRAPH_DEBUG(
            "setting input ", i, " to type ", *n->ty(attr::profiled_type));
        dni->setType(n->ty(attr::profiled_type));
      } else if (dni->node()->kind() == prim::DifferentiableGraph) {
        // The profiling node might have been absorbed in a preceding
        // differentiable graph and thus not (not ideal for fusing either),
        // see TestAutodiffSubgraphSlicing.test_does_not_create_cycles.
        // Alternatives to this special casing could be specializing the types
        // before autodiff or duplicating profile nodes for autodiff outputs
        // but that should be done while creating subgraphs and would be
        // a mess.
        // XXX TODO: revisit the alternatives
        Value* o = dni->node()->g(attr::Subgraph)->outputs().at(dni->offset());
        if (o->node()->kind() == prim::profile) {
          dni->setType(o->node()->ty(attr::profiled_type));
        }
      }

      // Propagate the requires_grad property to inputs
      // A RequiresGrad check gets added (insertTypeGuard, below)
      // so requires_grad is guaranteed to match for the inputs;
      // but other properties are not guaranteed to match
      auto requires_grad = dni->type()->expectRef<TensorType>().requiresGrad();
      gi[i]->setType(ty->withRequiresGrad(requires_grad));

      // we check if the optional is defined
      all_inputs_seen &= (dni->type()->cast<TensorType>() != TensorType::get());
    }
  }
  if (all_inputs_seen) {
    // we may have seen both true and false for requires_grad. In this case
    // we guard with true here and the other case is in the fallback. This
    // will give us trouble when we get "alternating patterns" of gradients
    // of two inputs, but so it is. An alternative could be to look into
    // the individual requires_grad seen in the profiling record.
    insertTypeGuard(
        dnode,
        [](const TensorTypePtr& t) {
          return TensorType::get()->withRequiresGrad(
              t->requiresGrad().value_or(true));
        },
        prim::RequiresGradCheck);
    return true;
  } else {
    // we inline the differentiable graph as a fallback
    // ideally we would set this up for re-profiling
    UpdateDifferentiableGraphRequiresGrad(
        dnode->g(attr::Subgraph), std::nullopt);
    SubgraphUtils::unmergeSubgraph(dnode);
    return false;
  }
}

void runNooptPassPipeline(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG("Before Inliner (beginning of runNooptPassPipeline)\n", *graph);
  Inline(*graph);
  GRAPH_DEBUG("After Inline, Before NoGrad\n", *graph);
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

static void runPreAutodiffPassPipeline(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG(
      "Before InsertGuards (beginning of runPreAutodiffPassPipeline)\n",
      *graph);

  LowerGradOf(*graph);
  GRAPH_DEBUG("After LowerGradOf, before specializeAutogradZero\n", *graph);

  specializeAutogradZero(graph);
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

FusionBehavior ProfilingGraphExecutorImpl::getCurrentBehavior(
    size_t remaining_depth) {
  size_t curr_depth = 0;
  for (int i = static_cast<int>(fusion_strategy_.size()) - 1; i >= 0; i--) {
    curr_depth += fusion_strategy_[i].second;
    if (remaining_depth <= curr_depth) {
      return fusion_strategy_[i].first;
    }
  }
  // should never get here
  TORCH_WARN("Strategy changed mid-invocation, NYI");
  return FusionBehavior::STATIC;
}

void ProfilingGraphExecutorImpl::runNoGradOptimizations(
    std::shared_ptr<Graph>& graph,
    size_t remaining_bailout_depth) {
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
    GRAPH_DEBUG("After LowerSimpleTuples\n", *graph);

    if (tensorExprFuserEnabled()) {
      // Remove prim::profile nodes and embed the profile info directly in the
      // IR in value types. We're doing such transformation as optimizations
      // that try to merge/fuse nodes in the graph (e.g. BatchMM and GraphFuser)
      // work worse in the presence of intermittent prim::profile nodes.
      // Optimizations relying on the type info are also responsible for
      // inserting proper type checks. Once we're done with these optimizations
      // we will wipe the tensor type information from the IR, so that it's not
      // accidentally used by any other pass.
      RemoveProfileNodesAndSpecializeTypes(graph);
      GRAPH_DEBUG(
          "After RemoveProfileNodesAndSpecializeTypes, before BatchMM\n",
          *graph);
      // Rewrite subgraphs with many MMs into expressions that batch them.
      BatchMM(graph);
      GRAPH_DEBUG("After BatchMM, before Fusion\n", *graph);
      auto min_size = getFusionGroupInlining() ? 2 : 1;
      bool dyn_shapes = getCurrentBehavior(remaining_bailout_depth) ==
          FusionBehavior::DYNAMIC;
      FuseTensorExprs(graph, min_size, /* composed op*/ false, dyn_shapes);
      GRAPH_DEBUG("After Fusion, before customPostPasses\n", *graph);
    } else {
      // Rewrite subgraphs with many MMs into expressions that batch them.
      BatchMM(graph);
      GRAPH_DEBUG("After BatchMM, before Fusion\n", *graph);

      FuseGraph(graph, true);
      GRAPH_DEBUG("After Fusion, before customPostPasses\n", *graph);
    }

    // Run custom post-fusion passes
    for (const auto& passPair : getCustomPostPasses()) {
      passPair.first(graph);
    }
    GRAPH_DEBUG(
        "After customPostPasses, before RemoveTensorTypeSpecializations \n",
        *graph);
    RemoveTensorTypeSpecializations(graph);
    GRAPH_DEBUG("After RemoveTensorTypeSpecializations\n", *graph);
  }
  GRAPH_DEBUG("End of runNoGradOptimizations\n");
}

void ProfilingGraphExecutorImpl::runProfilingOptimizations(
    std::shared_ptr<Graph>& copy,
    size_t remaining_bailout_depth) {
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
      GRAPH_DEBUG("Optimizing diff node ", idx, " in ", *copy);
      if (!guardDifferentiableGraph(dnode)) {
        // if we cannot guard (because of inputs without profiling information),
        // we re-inline the subgraph and remove the differentiable node
        GRAPH_DEBUG("Could not guardDifferentiableGraph ", idx, " in ", *copy);
        idx++;
        continue;
      }
      GRAPH_DEBUG("After guardDifferentiableGraph:\n", *copy);
      auto diff_graph = std::move(dnode->g(attr::Subgraph));
      Gradient gradient = differentiate(diff_graph);
      RemoveTensorTypeSpecializations(gradient.f);
      ProfilingRecord::removeProfilingNodes(gradient.f->block());
      GRAPH_DEBUG("Forward graph:\n", *(gradient.f));
      GRAPH_DEBUG("Backward graph:\n", *(gradient.df));
      // just like inside autograd.Functions, the forward of a differentiable
      // graph is essentially in a torch.no_grad context.
      UpdateDifferentiableGraphRequiresGrad(gradient.f, false);
      GRAPH_DEBUG("After UpdateDifferentiableGraphRequiresGrad ", *gradient.f);
      // replaces fallback graphs inserted by TE Fuser
      replaceFallbackGraphWithFallbackFunction(gradient.f->block());
      packGradient(gradient, dnode);
      GRAPH_DEBUG("Finished optimizing diff node ", idx++);
    }
    InlineAutodiffSubgraphs(
        copy,
        getAutodiffSubgraphInlining() ? autodiffSubgraphNodeThreshold : 1);
    replaceFallbackGraphWithFallbackFunction(copy->block());
    ProfilingRecord::removeProfilingNodes(copy->block());
    GRAPH_DEBUG(
        "After InlineAutodiffSubgraphs and Removing Profiling Nodes\n", *copy);
  } else {
    runNoGradOptimizations(copy, remaining_bailout_depth);
  }
  EliminateDeadCode(copy);
  GRAPH_DEBUG("After runProfilingOptimizations:\n", *copy);
}

void ProfilingGraphExecutorImpl::runProfilingInsensitiveOptimizations(
    std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG(
      "Before inlining (beginning of runProfilingInsensitiveOptimizations)\n",
      *graph);
  // TODO: maybe this can go later in pipeline / directly in autodiff forward
  // creation
  if (getGraphExecutorOptimize()) {
    Inline(*graph);
  }
  GRAPH_DEBUG("After inlining, before ClearProfilingInformation\n", *graph);
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
    : GraphExecutorImplBase(graph, std::move(function_name)) {
  fusion_strategy_ = getFusionStrategy();
}

size_t ProfilingGraphExecutorImpl::getInstantiatedBailoutDepth() {
  // Initialize bailout_depth from command-line flag.
  size_t depth = 0;
  for (const auto& pair : fusion_strategy_) {
    depth += pair.second;
  }
  return depth;
}

const ExecutionPlan& ProfilingGraphExecutorImpl::getOptimizedPlanFor(
    Stack& stack,
    std::optional<size_t> remaining_bailout_depth) {
  GRAPH_DEBUG("Running ProfilingGraphExecutorImpl ", this);

  // TODO: instantiate simple executor when getProfilingMode() is false
  // no opt mode
  if (!getGraphExecutorOptimize() || !getProfilingMode()) {
    if (!fallback_plan_) {
      auto copy = graph->copy();
      GRAPH_DEBUG(
          "Before LowerGradOf (beginning of runNooptPassPipeline)\n", *graph);
      LowerGradOf(*copy);
      GRAPH_DEBUG("After LowerGradOf, before RemoveExpands\n", *graph);
      RemoveExpands(copy);
      fallback_plan_ = ExecutionPlan(copy, function_name_);
      GRAPH_DUMP("NoOpt Graph: ", copy);
    }
    return *fallback_plan_;
  }

  // if tensorExprFuserEnabled() returns true we need to persist the very first
  // time ProfilingGraphExecutorImpl is called, so we can update it correctly
  // for fallback functions in ProfilingGraphExecutorImpl Else,
  // getPlanFor(remaining_bailout_depth) is corrected and persisted by the Code
  // object in interpreter.
  if (!remaining_bailout_depth_.has_value() || !tensorExprFuserEnabled()) {
    if (remaining_bailout_depth.has_value()) {
      remaining_bailout_depth_ = *remaining_bailout_depth;
    } else {
      remaining_bailout_depth_ = getInstantiatedBailoutDepth();
    }
  }

  // simple executor
  if (*remaining_bailout_depth_ == 0) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy);
    GRAPH_DUMP("Optimized SimpleExecutor Graph: ", copy);
    optimized_plan_ = ExecutionPlan(copy, function_name_);
    time_optimized_plan_created_ = getNowInSecs();
    return *optimized_plan_;
  }

  bool profiling_record_created_in_this_call = false;
  // if a profiling graph hasn't been created yet
  if (!pr_) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy);
    pr_ = ProfilingRecord::instrumentGraph(copy);
    profiling_record_created_in_this_call = true;
    // `InsertProfileNodesForSpecializeAutogradZero` profiles a definition vs a
    // use and it doesn't expect any profile nodes between a graph input and its
    // consumer, `aten::_grad_sum_to_size`. This means we need to run it first,
    // before any other pass that could insert `prim::iprofile_value` node on
    // `aten::_grad_sum_to_size` input.
    InsertProfileNodesForSpecializeAutogradZero(pr_.get());
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
  runProfilingOptimizations(copy, *remaining_bailout_depth_);
  // replaces a fallback graph inserted by
  // specialize_autogradzero if one exists
  replaceFallbackGraphWithFallbackFunction(copy->block());
  runFinalOptimizations(copy);
  CheckStrictFusion(copy);
  GRAPH_DUMP("Optimized Graph: ", copy);
  optimized_plan_ = ExecutionPlan(copy, function_name_);
  time_optimized_plan_created_ = getNowInSecs();
  // If the profiled graph was created in this call, then we can release it
  // right.
  if (FLAGS_torch_jit_release_profiling_graph_after_optimization &&
      profiling_record_created_in_this_call) {
    clearTheGraphCompilationIntermediateGraphs();
  }
  return *optimized_plan_;
}

const ExecutionPlan& ProfilingGraphExecutorImpl::getPlanFor(
    Stack& stack,
    std::optional<size_t> remaining_bailout_depth) {
  std::lock_guard<std::mutex> lock(compile_mutex);

  // IMPORTANT: This is a hot path of calling a torchscript function. Try not to
  // add any code above this.
  if (optimized_plan_) {
    if (FLAGS_torch_jit_release_profiling_graph_after_optimization &&
        !is_graph_extra_memory_released_) {
      int32_t now = getNowInSecs();
      if ((now - time_optimized_plan_created_) >
          FLAGS_torch_jit_release_profiling_graph_delay_in_seconds) {
        clearTheGraphCompilationIntermediateGraphs();
      }
    }
    return *optimized_plan_;
  }
  // if depth is not set, use
  return getOptimizedPlanFor(stack, remaining_bailout_depth);
}

GraphExecutorState ProfilingGraphExecutorImpl::getDebugState() {
  GraphExecutorState state;
  TORCH_INTERNAL_ASSERT(optimized_plan_);
  auto opt_plan = *optimized_plan_;
  state.execution_plans.emplace(ArgumentSpec{0, 0}, opt_plan);
  return state;
}

static Node* insertFallbackFunctionCall(
    Graph* graph,
    GraphFunction* func,
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

static GraphFunction* createFallbackPathFunction(
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
      for (const auto i : c10::irange(function_call->outputs().size())) {
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

void ProfilingGraphExecutorImpl::runFinalOptimizations(
    std::shared_ptr<Graph>& graph) {
  AddIfThenElseOp(graph);
}

void ProfilingGraphExecutorImpl::debugFlushCompilationCache() {
  std::lock_guard<std::mutex> lock(compile_mutex);
  pr_.reset();
  fallback_plan_.reset();
  profiling_plan_.reset();
  optimized_plan_.reset();
  // prevent memory leaks
  fallback_functions_.clear();
  remaining_bailout_depth_.reset();
  // TODO - would be nice to have it initialized in subsequent use
  fusion_strategy_ = getFusionStrategy();
  time_optimized_plan_created_ = 0;
  is_graph_extra_memory_released_ = false;
}

void ProfilingGraphExecutorImpl::clearTheGraphCompilationIntermediateGraphs() {
  is_graph_extra_memory_released_ = true;
  profiling_plan_.reset();
  fallback_plan_.reset();
  graph.reset();
  pr_.reset();
}

} // namespace torch::jit
