#include <torch/csrc/jit/graph_executor.h>

#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/graph_executor_impl.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/passes/canonicalize_ops.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/inplace_check.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/quantization.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/profiling_record.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/jit/script/logging.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

// for debugging it is helpful to be able to force autodiff subgraphs
// to be created, to check their correctness, even when the
// size of the of the subgraph is too small to be profitable.
thread_local bool autodiff_subgraph_inlining = true;
void debugSetAutodiffSubgraphInlining(bool state) {
  autodiff_subgraph_inlining = state;
}

bool getAutodiffSubgraphInlining() {
  return autodiff_subgraph_inlining;
}

thread_local std::weak_ptr<Graph> last_executed_optimized_graph;
std::shared_ptr<Graph> lastExecutedOptimizedGraph() {
  return last_executed_optimized_graph.lock();
}

void GraphExecutorImplBase::run(Stack& stack) {
  TORCH_CHECK(
      stack.size() >= num_inputs,
      "expected ",
      num_inputs,
      " inputs, but got only ",
      stack.size());

  C10_LOG_API_USAGE_ONCE("torch.graph_executor.run");
  logging::getLogger()->addStatValue(
      logging::runtime_counters::GRAPH_EXECUTOR_INVOCATIONS, 1.0);

  ExecutionPlan plan = getPlanFor(stack);
  InterpreterState(plan.code).run(stack);
  last_executed_optimized_graph = plan.graph;
}

// a Graph can be created via tracing, or via a language-based frontend
// GraphExecutor runs it. It can run the same graph on many different sizes
// and different requires_grad states, and handles specializations for each
// situation. GraphExecutor is completely unaware of tracing or module
// parameters to keep the tracing concerns separated.
struct GraphExecutorImpl : public GraphExecutorImplBase {
  GraphExecutorImpl(const std::shared_ptr<Graph>& graph)
      : GraphExecutorImplBase(graph), arg_spec_creator_(*graph) {
    logging::getLogger()->addStatValue(
        logging::runtime_counters::GRAPH_EXECUTORS_CONSTRUCTED, 1.0);
  }

  ExecutionPlan getPlanFor(Stack& stack) override {
    return getGraphExecutorOptimize() ? getOrCompile(stack)
                                      : getOrCompileFallback();
  }

  GraphExecutorState getDebugState() override {
    GraphExecutorState state;
    state.graph = graph.get();
    if (fallback) {
      state.fallback = fallback;
    }
    for (auto& entry : plan_cache) {
      state.execution_plans.emplace(entry.first, entry.second);
    }
    return state;
  }

 protected:
  friend struct GraphExecutor;

  const ExecutionPlan& getOrCompileFallback() {
    std::lock_guard<std::mutex> lock(compile_mutex);
    if (!fallback) {
      auto graph_ = graph->copy();
      runRequiredPasses(graph_);
      fallback = ExecutionPlan(graph_);
    }
    return fallback;
  }

  const ExecutionPlan& getOrCompile(const Stack& stack) {
    // outside lock guard, to minimize the time holding the lock on the fast
    // path ArgumentSpec even computes its hashCode here.
    ArgumentSpec spec =
        arg_spec_creator_.create(autograd::GradMode::is_enabled(), stack);
    {
      std::lock_guard<std::mutex> lock(compile_mutex);
      auto it = plan_cache.find(spec);
      if (it != plan_cache.end()) {
        logging::getLogger()->addStatValue(
            logging::runtime_counters::EXECUTION_PLAN_CACHE_HIT, 1.0);
        return it->second;
      }
      auto plan = compileSpec(spec);
      auto r = plan_cache.emplace(std::move(spec), std::move(plan));
      logging::getLogger()->addStatValue(
          logging::runtime_counters::EXECUTION_PLAN_CACHE_MISS, 1.0);
      return r.first->second;
    }
  }

  ExecutionPlan compileSpec(const ArgumentSpec& spec) {
    auto opt_graph = graph->copy();
    SOURCE_DUMP("Optimizing the following function:", opt_graph);
    arg_spec_creator_.specializeTypes(*opt_graph, spec);

    // Phase 0. Inline functions, then clean up any artifacts that the inliner
    //          left in that may inhibit optimization
    Inline(*opt_graph);
    specializeAutogradZero(*opt_graph);
    LowerSimpleTuples(opt_graph);
    ConstantPooling(opt_graph);

    // Phase 1. Specialize to input definedness (this is very important for
    //          gradient graphs), and run required passes to bring the graph
    //          to an executable form.
    runRequiredPasses(opt_graph);

    // Phase 2. Propagate detailed information about the spec through the
    //          graph (enabled more specializations in later passes).
    //          Shape propagation sometimes depends on certain arguments being
    //          constants, and constant propagation doesn't need shape
    //          information anyway, so it's better to run it first.
    ConstantPropagation(opt_graph);
    PropagateInputShapes(opt_graph);
    PropagateRequiresGrad(opt_graph);

    // Phase 3. Run differentiable optimizations (i.e. simple graph rewrites
    //          that we can still execute using autograd).
    runOptimization(opt_graph);

    // Phase 4. If this graph will be differentiated, we need to slice out the
    //          symbolically differentiable subgraphs for further optimizations.
    // Phase 5. Apply non-differentiable optimizations to the graphs we've found
    //          (or the whole grpah if we know we won't need its derivative).
    if (needsGradient(opt_graph)) {
      auto diff_nodes = CreateAutodiffSubgraphs(
          opt_graph,
          autodiff_subgraph_inlining ? autodiffSubgraphNodeThreshold : 1);
      for (Node* dnode : diff_nodes) {
        auto diff_graph = std::move(dnode->g(attr::Subgraph));
        Gradient gradient = differentiate(diff_graph);
        // Run post differentiation optimizations, Autodiff will replace some
        // parts of graph with new graph, these new graphs usually consists of
        // control flows and miss shape information on nodes, so we run shape
        // prop and differentiable optimizations to ensure the graph is
        // optimized
        PropagateInputShapes(gradient.f);
        runOptimization(gradient.f);
        // run non diff optimization on the forward graph
        runNondiffOptimization(gradient.f);
        packGradient(gradient, dnode);
      }
      InlineAutodiffSubgraphs(
          opt_graph,
          autodiff_subgraph_inlining ? autodiffSubgraphInlineThreshold : 1);
    } else {
      runNondiffOptimization(opt_graph);
    }
    // Make sure there are no leftovers from any passes.
    EliminateDeadCode(opt_graph);
    return ExecutionPlan(opt_graph);
  }

  ~GraphExecutorImpl() override = default;

  ArgumentSpecCreator arg_spec_creator_;
  // Populated only when optimize is false (and in that case plan_cache will be
  // unused). The compiled version of graph.
  ExecutionPlan fallback;

  // Mapping from argument configurations to optimized versions of the graph
  // that are specialized to the spec.
  std::unordered_map<ArgumentSpec, ExecutionPlan> plan_cache;
};

GraphExecutor::GraphExecutor(std::shared_ptr<Graph> graph)
    : pImpl(
          getProfilingMode() ? dynamic_cast<GraphExecutorImplBase*>(
                                   new ProfilingGraphExecutorImpl(graph))
                             : dynamic_cast<GraphExecutorImplBase*>(
                                   new GraphExecutorImpl(graph))) {}

void GraphExecutor::run(Stack& inputs) {
  return pImpl->run(inputs);
}

ExecutionPlan GraphExecutor::getPlanFor(Stack& inputs) {
  return pImpl->getPlanFor(inputs);
}

std::shared_ptr<Graph> GraphExecutor::graph() const {
  return pImpl->graph;
}

GraphExecutorState GraphExecutor::getDebugState() {
  return pImpl->getDebugState();
}

void runRequiredPasses(const std::shared_ptr<Graph>& g) {
  LowerGradOf(*g);
  // implicit inserted expand nodes are not necessarily always valid
  // when used inside script methods that might have unstable shapes
  // we remove the implicitly created ones, and have shape analysis
  // add valid expand nodes when the shapes are stable
  RemoveExpands(g);
  CanonicalizeOps(g);
  EliminateDeadCode(g);
}

void packGradient(const Gradient& gradient, Node* dnode) {
  AT_ASSERT(dnode->kind() == prim::DifferentiableGraph);
  dnode->g_(attr::Subgraph, gradient.f)
      ->g_(attr::ReverseSubgraph, gradient.df)
      ->i_(attr::f_real_outputs, gradient.f_real_outputs)
      ->is_(attr::df_input_vjps, fmap<int64_t>(gradient.df_input_vjps))
      ->is_(
          attr::df_input_captured_inputs,
          fmap<int64_t>(gradient.df_input_captured_inputs))
      ->is_(
          attr::df_input_captured_outputs,
          fmap<int64_t>(gradient.df_input_captured_outputs))
      ->is_(attr::df_output_vjps, fmap<int64_t>(gradient.df_output_vjps));
}

static bool mayIntroduceGradient(const Block* b) {
  for (const Node* n : b->nodes()) {
    if (n->kind() == prim::PythonOp)
      return true;
    for (const Block* bb : n->blocks()) {
      if (mayIntroduceGradient(bb))
        return true;
    }
  }
  return false;
}

bool needsGradient(const std::shared_ptr<const Graph>& graph) {
  if (!autograd::GradMode::is_enabled())
    return false;
  if (mayIntroduceGradient(graph->block()))
    return true;
  for (const Value* input : graph->inputs()) {
    if (input->type()->requires_grad())
      return true;
  }
  return false;
}

void runNondiffOptimization(std::shared_ptr<Graph>& graph) {
  // run custom passes that different backends can register
  for (const auto& pass : getCustomPasses()) {
    pass(graph);
  }
  // decomposition pass, decompose certain ops that will be used in the
  // following passes (like batchmm and jit fusion)
  DecomposeOps(graph);

  // TupleConstruct / TupleUnpack pairs can still be present at this point
  // and must be removed for fusion.
  LowerSimpleTuples(graph);

  // Rewrite subgraphs with many MMs into expressions that batch them.
  BatchMM(graph);

  // Fuse the dequant - op - quant patterns into quantized ops
  QuantFusion(graph);

  FuseGraph(graph);
}

void runOptimization(std::shared_ptr<Graph>& graph) {
  // Basic graph preprocessing to eliminate noise.
  EliminateDeadCode(graph);
  EliminateCommonSubexpression(graph);
  ConstantPooling(graph);

  PeepholeOptimize(graph);
  ConstantPropagation(graph);

  // Unroll small loops, and eliminate expressions that are the same at every
  // iteration.
  UnrollLoops(graph);
  EliminateCommonSubexpression(graph);

  CheckInplace(graph);
}

} // namespace jit
} // namespace torch
