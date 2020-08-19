#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
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
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include "ATen/core/interned_strings.h"
#include "jit/ir/ir.h"
#include "jit/passes/inliner.h"
#include <aten/src/ATen/core/jit_type.h>

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

    for (auto o : n->outputs()) {
      if (auto ptt = o->type()->cast<TensorType>()) {
        if (!ptt->requiresGrad() || *ptt->requiresGrad()) {
          return true;
        }
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

void removeProfilingNodes(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::profile) {

      if (it->outputs().size()) {
        it->output()->replaceAllUsesWith(it->input());
      }
      it.destroyCurrent();
    } else {
      for (Block* ib : it->blocks()) {
        removeProfilingNodes(ib);
      }
    }
  }
}

void ProfilingGraphExecutorImpl::runProfilingOptimizations(
    std::shared_ptr<Graph>& copy) {
  if (!getGraphExecutorOptimize()) {
    LowerGradOf(*copy);
    runRequiredPasses(copy);
    return;
  }

  GRAPH_DUMP("Before running optimizations:", copy);
  runOptimization(copy, true);
  GRAPH_DUMP("Before removing profiling nodes:", copy);
  GRAPH_DUMP("After fusion:", copy);
  LowerGradOf(*copy);
  GRAPH_DUMP("After InsertBailOuts: ", copy);
  specializeAutogradZero(*copy);
  // this should remove grad_sum_to_size
  PeepholeOptimize(copy);

  runRequiredPasses(copy);
  //PeepholeOptimize(copy);
  ConstantPropagation(copy);
  runOptimization(copy, false);

  if (needsGradientInProfilingMode(copy->block())) {
    GRAPH_DEBUG("Running CreateAutodiffSubgraphs");
    auto diff_nodes = CreateAutodiffSubgraphs(
        copy,
        getAutodiffSubgraphInlining() ? autodiffSubgraphNodeThreshold : 1);
    for (Node* dnode : diff_nodes) {
      auto diff_graph = std::move(dnode->g(attr::Subgraph));
      Gradient gradient = differentiate(diff_graph);
      runOptimization(gradient.f);
      // run non diff optimization on the forward graph
      runNondiffOptimization(gradient.f, true);
      replaceFallbackGraphWithFallbackFunction(gradient.f->block());
      packGradient(gradient, dnode);
    }
    InlineAutodiffSubgraphs(
        copy,
        /* getAutodiffSubgraphInlining() ? autodiffSubgraphInlineThreshold :*/ 1);

  } else {
    GRAPH_DEBUG("Running no needsGradientInProfilingMode version");
    runNondiffOptimization(copy, true);
  }
  removeProfilingNodes(copy->block());
  EliminateDeadCode(copy);
  GRAPH_DUMP("Optimized Graph : ", copy);
}

void ProfilingGraphExecutorImpl::runProfilingInsensitiveOptimizations(
    std::shared_ptr<Graph>& copy) {
  Inline(*copy);
  ClearProfilingInformation(copy);
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
  LowerSimpleTuples(copy);
  CheckInplace(copy);
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
    return *optimized_plan_;
  }

  // simple executor
  if (remaining_bailout_depth == 0) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy);
    GRAPH_DUMP("Optimized SimpleExecutor Graph : ", copy);
    optimized_plan_ = ExecutionPlan(copy, function_name_);
    return *optimized_plan_;
  }

  // if a profiling graph hasn't been created yet
  if (!pr_) {
    auto copy = graph->copy();
    removeProfilingNodes(copy->block());
    runProfilingInsensitiveOptimizations(copy);
    // if (remaining_bailout_depth == getBailoutDepth()) {	
    //   PeelProfilingLoops(copy);	
    // }
    pr_ = ProfilingRecord::instrumentGraph(copy);
    auto pr_copy = pr_->graph()->copy();
    GRAPH_DUMP("Profiled Graph: ", pr_copy);
    profiling_plan_ = ExecutionPlan(pr_copy, function_name_);
    // fall-through
  }

  // profile until a graph is ready
  if (!pr_->ready()) {
        auto pr_copy = pr_->graph()->copy();
    GRAPH_DUMP("Profiled Graph: ", pr_copy);
    return ExecutionPlan(pr_copy, function_name_);
  }

  auto copy = pr_->graph()->copy();
  GRAPH_DUMP("before runProfilingOptimizations: ", copy);
  runProfilingOptimizations(copy);
  replaceFallbackGraphWithFallbackFunction(copy->block());
  GRAPH_DUMP("After replaceFallbackGraphWithFallbackFunction : ", copy);
  // cache
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

void ProfilingGraphExecutorImpl::registerFallbackFunctions(Block* b) {
  for (auto n : b->nodes()) {
    if (n->kind() == prim::Constant && n->hasAttribute(Symbol::attr("fallback"))) {
      if (auto ftype = n->output()->type()->cast<FunctionType>()) {
        Stack s;
        ftype->function()->get_executor().getPlanFor(s, bailout_depth - 1);
        fallback_functions_.emplace_back(ftype->function());
      }
    } else {
      for (auto ib: n->blocks()) {
        registerFallbackFunctions(ib);
      }
    }
  }
}

Node* createFallbackGraph(Block* b, ArrayRef<Value*> inputs, Graph* g) {

  auto graph = std::make_shared<Graph>();
  auto value_map = [](Value* v) { return v; };
  graph->block()->cloneFrom(b, value_map);

  auto fallback = g->create(prim::FallbackGraph, inputs, b->outputs().size());
  fallback->g_(attr::Subgraph, graph);

  for (size_t i = 0; i < b->outputs().size(); i++) {
    fallback->output(i)->setType(b->outputs()[i]->type());
    fallback->output(i)->copyMetadata(b->outputs()[i]);
  }
  return fallback;
}

void ProfilingGraphExecutorImpl::replaceFallbackGraphWithFallbackFunction(Block* b) {

  Stack s;
  for (auto it = b->nodes().begin(); it != b->nodes().end(); ) {
    if (it->kind() == prim::FallbackGraph) {
      auto fallback_func = createFallbackPathFunction(it->g(attr::Subgraph)->block(), "fallback_function");
      fallback_func->get_executor().getPlanFor(s, bailout_depth - 1);
      fallback_functions_.emplace_back(fallback_func);
      WithInsertPoint wip {*it};
      auto function_call = insertFallbackFunctionCall(b->owningGraph(), fallback_func, it->inputs());
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

Function* createFallbackPathFunction(Block* b, const std::string& function_name) {
    
    auto value_map = [](Value* v) { return v; };
    auto graph = std::make_shared<Graph>();
    graph->block()->cloneFrom(b, value_map);

    auto otypes = c10::fmap(graph->return_node()->inputs(), [](Value* v) {return v->type(); });
    auto tuple_type = TupleType::create(otypes);
    auto return_tuple = graph->createTuple(graph->return_node()->inputs());

    // {
    //   WithInsertPoint wip(*graph->block()->nodes().begin()); 
    //   auto debug_print_cnst = graph->insertConstant(IValue{std::string("graph")});
    //   auto print_stmt = graph->insert(prim::Print, {debug_print_cnst});
    // }
    
    graph->appendNode(return_tuple);
    for (int i = graph->outputs().size() - 1; i >= 0; i--) {
      graph->eraseOutput(i);
    }
    graph->registerOutput(return_tuple->output());
    //GRAPH_DUMP("graph", graph);
    return new GraphFunction(function_name, graph, nullptr);
}

Node* insertFallbackFunctionCall(Graph* graph, Function* func, ArrayRef<Value*> inputs) {
    auto tuple_type = func->graph()->return_node()->input(0)->type();
    Value* fn_constant = graph->insertNode(graph->create(prim::Constant))
                           ->s_(attr::name, func->name())
                           ->i_(Symbol::attr("fallback"), 1)
                           ->output()
                           ->setType(FunctionType::create(func));
    std::vector<Value*> func_call_inputs = {fn_constant};
    func_call_inputs.insert(func_call_inputs.end(), inputs.begin(), inputs.end());
    Value* result = graph->insertNode(graph->create(prim::CallFunction, func_call_inputs))
                      ->output()
                      ->setType(tuple_type);

    auto fun_unpack_tuple = graph->insertNode(graph->createTupleUnpack(result));
    return fun_unpack_tuple;
}

} // namespace jit
} // namespace torch
