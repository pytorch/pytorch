#include <torch/csrc/jit/graph_executor.h>

#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/autodiff.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/resource_guard.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/passes/canonicalize_ops.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/inplace_check.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_undef.h>
#include <torch/csrc/jit/symbolic_variable.h>
#include <torch/csrc/jit/tracer.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/script/compiler.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

namespace {

using tensor_list = std::vector<at::Tensor>;
using Variable = autograd::Variable;
using autograd::variable_list;

struct ExecutionPlan {
  ExecutionPlan() = default;
  ExecutionPlan(std::shared_ptr<Graph> graph)
      : code(graph), graph(std::move(graph)) {}

  void run(Stack& stack) const {
    return InterpreterState(code).run(stack);
  }

  operator bool() const {
    return static_cast<bool>(graph);
  }

  ExecutionPlanState getDebugState() {
    ExecutionPlanState state;
    state.code = &code;
    state.graph = graph.get();
    return state;
  }

  Code code;
  std::shared_ptr<Graph> graph;
};

struct DifferentiableGraphBackward : public autograd::Function {
  DifferentiableGraphBackward(GraphExecutor executor, size_t capture_size)
      : executor(std::move(executor)) {
    is_var_capture.reserve(capture_size);
    var_captures.reserve(capture_size);
    ivalue_captures.reserve(capture_size);
  }

  variable_list apply(variable_list&& inputs) override {
    Stack stack;
    stack.reserve(is_var_capture.size() + inputs.size());
    stack.insert(
        stack.end(),
        std::make_move_iterator(inputs.begin()),
        std::make_move_iterator(inputs.end()));
    auto var_capture_it = var_captures.begin();
    auto ivalue_capture_it = ivalue_captures.begin();
    for (bool is_var : is_var_capture) {
      if (is_var) {
        stack.emplace_back(var_capture_it->unpack(this->shared_from_this()));
        ++var_capture_it;
      } else {
        stack.push_back(*ivalue_capture_it);
        ++ivalue_capture_it;
      }
    }

    executor.run(stack);
    AT_ASSERT(stack.size() == num_outputs());

    variable_list outputs;
    outputs.reserve(num_outputs());
    for (size_t i = 0; i < num_outputs(); ++i) {
      if (should_compute_output(i)) {
        auto output = std::move(stack[i]).toTensor();
        const auto& edge = next_edge(i);
        if (output.defined()) {
          outputs.emplace_back(std::move(output));
        } else if (edge.is_valid()) {
          outputs.emplace_back(
              edge.function->input_metadata(edge.input_nr).zeros_like());
        } else {
          outputs.emplace_back();
        }
      } else {
        outputs.emplace_back();
      }
    }
    return outputs;
  }

  void capture(const IValue& val, bool is_output) {
    const bool is_tensor = val.isTensor();
    is_var_capture.push_back(is_tensor);
    if (is_tensor) {
      var_captures.emplace_back(Variable(val.toTensor()), is_output);
    } else {
      ivalue_captures.push_back(val);
    }
  }

 private:
  friend struct ExecutionPlan;
  GraphExecutor executor;

  // INVARIANT: is_var_capture.size() == var_captures.size() +
  // ivalue_captures.size()
  std::vector<bool> is_var_capture;
  std::vector<autograd::SavedVariable> var_captures;
  std::vector<IValue> ivalue_captures;
};

// an optimized way of executing the subgraph computed directly on
// tensors rather than Variables.
// This will unwrap Variables, run the plan, and re-wrap them.
// It can optionally also have a gradient which is hooked up
// to the output Variables if present.
struct DifferentiableGraphOp {
  DifferentiableGraphOp(Gradient grad)
      : f(grad.f),
        grad(std::move(grad)),
        grad_executor(this->grad.df),
        num_inputs(this->grad.f->inputs().size()),
        num_outputs(this->grad.f->outputs().size()) {}

  // XXX: keep in mind that stack can be larger than the inputs we need!
  int operator()(Stack& stack) const {
    auto grad_fn = std::make_shared<DifferentiableGraphBackward>(
        grad_executor,
        grad.df_input_captured_inputs.size() +
            grad.df_input_captured_outputs.size());

    {
      auto inputs = last(stack, num_inputs);
      // hook up the outputs of df to the gradient functions of the inputs that
      // require gradients
      for (auto idx : grad.df_output_vjps) {
        auto v = Variable(inputs[idx].toTensor());
        grad_fn->add_next_edge(
            v.defined() ? v.gradient_edge() : autograd::Edge{});
      }
      captureInputs(*grad_fn, inputs);
    }

    detachVariables(stack);
    InterpreterState(f).run(stack);

    {
      auto outputs = last(stack, num_outputs);
      // hookup the gradients for the output tensors that require gradients
      // to the inputs to our gradient function df
      // TODO - XXX - if any output is the same tensor multiple times, views
      // have to be setup here. We need to refactor autograd until it is safe
      // for tensors to be constructed without all the viewing infrastructure.
      // this is currently intentionally not done here so we can get an idea of
      // our perf before introducing overhead for correctness
      for (auto idx : grad.df_input_vjps) {
        // Note: we have to set this up in place, or we have to throw away and
        // reallocate variables that were already created in wrapTensors. We
        // should add an API for this.

        // XXX: undefined tensor syntax in autograd
        Variable output;
        if (!outputs[idx].isNone()) {
          output = outputs[idx].toTensor();
        }
        // NB: since our requires_grad setting is only a heuristic we might end
        // up wanting to differentiate through integral tensors, which is
        // generally a hard error in autograd.
        if (at::isFloatingType(output.type().scalarType())) {
          autograd::create_gradient_edge(output, grad_fn);
          output.set_requires_grad(true);
        } else {
          grad_fn->add_input_metadata(autograd::Function::undefined_input{});
        }
      }
      captureOutputs(*grad_fn, outputs);
      // drop the temporary outputs so that we return the same number of
      // outputs as if we were not also calculating gradient
      const size_t num_temporary_outputs = num_outputs - grad.f_real_outputs;
      stack.erase(stack.end() - num_temporary_outputs, stack.end());
    }
    return 0;
  }

 private:
  friend GraphExecutor* detail::getGradExecutor(Operation& op);

  void detachVariables(Stack& stack) const {
    // It would be nice to use an ArrayRef here, but unfortunately those can
    // only return const references, so we need to do a bunch of indexing
    // ourselves.
    const int64_t stack_size = stack.size();
    const int64_t stack_offset = stack_size - num_inputs;
    for (int64_t i = stack_offset; i < stack_size; ++i) {
      auto& v = stack[i];
      if (!v.isTensor())
        continue;
      auto t = std::move(v).toTensor();
      v = IValue{t.defined() ? autograd::as_variable_ref(t).detach()
                             : std::move(t)};
    }
  }
  // Capture (save) inputs that would be required to subsequently run backwards
  void captureInputs(
      DifferentiableGraphBackward& grad_fn,
      at::ArrayRef<IValue> inputs) const {
    for (size_t offset : grad.df_input_captured_inputs) {
      grad_fn.capture(inputs[offset], /*is_output*/ false);
    }
  }
  void captureOutputs(
      DifferentiableGraphBackward& grad_fn,
      at::ArrayRef<IValue> outputs) const {
    for (size_t offset : grad.df_input_captured_outputs) {
      grad_fn.capture(outputs[offset], /*is_output*/ true);
    }
  }

  Code f;
  Gradient grad;
  GraphExecutor grad_executor;

  const size_t num_inputs;
  const size_t num_outputs;
};

void packGradient(Gradient gradient, Node* dnode) {
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

Gradient getGradient(const Node* n) {
  AT_ASSERT(n->kind() == prim::DifferentiableGraph);
  Gradient grad;
  grad.f = n->g(attr::Subgraph);
  grad.df = n->g(attr::ReverseSubgraph);
  grad.f_real_outputs = n->i(attr::f_real_outputs);
  grad.df_input_vjps = fmap<size_t>(n->is(attr::df_input_vjps));
  grad.df_input_captured_inputs =
      fmap<size_t>(n->is(attr::df_input_captured_inputs));
  grad.df_input_captured_outputs =
      fmap<size_t>(n->is(attr::df_input_captured_outputs));
  grad.df_output_vjps = fmap<size_t>(n->is(attr::df_output_vjps));
  return grad;
}

} // anonymous namespace

RegisterOperators reg_graph_executor_ops(
    {Operator(prim::DifferentiableGraph, [](const Node* n) -> Operation {
      return DifferentiableGraphOp(getGradient(n));
    })});

namespace detail {

GraphExecutor* getGradExecutor(Operation& op) {
  if (auto diff_op = op.target<DifferentiableGraphOp>()) {
    return &diff_op->grad_executor;
  }
  return nullptr;
}

} // namespace detail

// a Graph can be created via tracing, or via a language-based frontend
// GraphExecutor runs it. It can run the same graph on many different sizes
// and different requires_grad states, and handles specializations for each
// situation. GraphExecutor is completely unaware of tracing or module
// parameters to keep the tracing concerns separated.
struct GraphExecutorImpl {
  static std::shared_ptr<Graph> prepareGraph(std::shared_ptr<Graph>& graph) {
    auto copy = graph->copy();
    EraseShapeInformation(copy);
    return copy;
  }

  static size_t countFlatInputs(const TypePtr& ptr) {
    if (auto optional_type = ptr->cast<OptionalType>()) {
      return countFlatInputs(optional_type->getElementType());
    }
    if (auto tuple_type = ptr->cast<TupleType>()) {
      size_t total = 0;
      for (auto& elem : tuple_type->elements()) {
        total += countFlatInputs(elem);
      }
      return total;
    }
    return 1;
  }

  static size_t countFlatInputs(const std::shared_ptr<Graph>& graph) {
    size_t total = 0;
    for (Value* input : graph->inputs()) {
      total += countFlatInputs(input->type());
    }
    return total;
  }

  inline bool hasMutableOperators(Block* block) {
    for (auto n : block->nodes()) {
      if (n->kind().is_aten() && n->schema().is_mutable())
        return true;
      for (auto b : n->blocks()) {
        if (hasMutableOperators(b))
          return true;
      }
    }
    return false;
  }

  GraphExecutorImpl(std::shared_ptr<Graph> graph, bool optimize)
      : graph(prepareGraph(graph)),
        // until we have correct alias analysis any use of mutable operators
        // disables all optimization
        optimize(optimize),
        num_inputs(this->graph->inputs().size()),
        num_flat_inputs(countFlatInputs(graph)),
        num_outputs(this->graph->outputs().size()) {}

  // entry point where execution begins
  void run(Stack& stack) {
    AT_CHECK(
        stack.size() >= num_inputs,
        "expected ",
        num_inputs,
        " inputs, but got only ",
        stack.size());

    if (tracer::isTracing()) {
      return runTraced(stack);
    }

    auto& execution_plan =
        optimize ? getOrCompile(stack) : getOrCompileFallback();
    return execution_plan.run(stack);
  }

  std::shared_ptr<Graph> graphFor(const Stack& stack) const {
    AT_ASSERT(stack.size() >= num_inputs);
    auto inputs = last(stack, num_inputs);
    ArgumentSpec spec(
        autograd::GradMode::is_enabled(), inputs, num_flat_inputs);

    if (!optimize) {
      AT_CHECK(fallback, "No graph found for given inputs");
      return fallback.graph;
    }

    auto it = plan_cache.find(spec);
    AT_CHECK(it != plan_cache.end(), "No graph found for given inputs");
    return it->second.graph;
  }

  GraphExecutorState getDebugState() {
    GraphExecutorState state;
    state.graph = graph.get();
    if (fallback) {
      state.fallback = fallback.getDebugState();
    }
    for (auto& entry : plan_cache) {
      state.execution_plans.emplace(entry.first, entry.second.getDebugState());
    }
    return state;
  }

  // This function should be used only for testing purposes
  void debugDisableAutodiffSubgraphInlining() {
    // Allow single-node autodiff subgraphs
    autodiffSubgraphNodeThreshold = 1;
    // Don't inline autodiff subgraphs into autograd functions
    autodiffSubgraphInlineThreshold = 1;
  }

 private:
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
    ArgumentSpec spec(
        autograd::GradMode::is_enabled(),
        last(stack, num_inputs),
        num_flat_inputs);
    {
      std::lock_guard<std::mutex> lock(compile_mutex);
      auto it = plan_cache.find(spec);
      if (it != plan_cache.end())
        return it->second;
      auto plan = compileSpec(spec);
      auto r = plan_cache.emplace(std::move(spec), std::move(plan));
      return r.first->second;
    }
  }

  ExecutionPlan compileSpec(const ArgumentSpec& spec) {
    auto opt_graph = graph->copy();
    setInputTypes(*opt_graph, spec);

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
    // that
    //          we can still execute using autograd).
    runOptimization(opt_graph, spec);

    // Phase 4. If this graph will be differentiated, we need to slice out the
    //          symbolically differentiable subgraphs for further optimizations.
    // Phase 5. Apply non-differentiable optimizations to the graphs we've found
    //          (or the whole grpah if we know we won't need its derivative).
    if (needsGradient(opt_graph)) {
      auto diff_nodes =
          CreateAutodiffSubgraphs(opt_graph, autodiffSubgraphNodeThreshold);
      for (Node* dnode : diff_nodes) {
        auto diff_graph = std::move(dnode->g(attr::Subgraph));
        Gradient gradient = differentiate(diff_graph);
        runNondiffOptimization(gradient.f);
        packGradient(gradient, dnode);
      }
      InlineAutodiffSubgraphs(opt_graph, autodiffSubgraphInlineThreshold);
    } else {
      runNondiffOptimization(opt_graph);
    }
    // Make sure there are no leftovers from any passes.
    EliminateDeadCode(opt_graph);
    return ExecutionPlan(opt_graph);
  }

  void runOptimization(
      std::shared_ptr<Graph>& graph,
      const ArgumentSpec& spec) {
    // Basic graph preprocessing to eliminate noise.
    EliminateDeadCode(graph);
    EliminateCommonSubexpression(graph);
    ConstantPooling(graph);

    PeepholeOptimize(graph);

    // Unroll small loops, and eliminate expressions that are the same at every
    // iteration.
    UnrollLoops(graph);
    EliminateCommonSubexpression(graph);

    // Rewrite subgraphs with many MMs into expressions that batch them.
    BatchMM(graph);

    CheckInplace(graph);
  }

  void runNondiffOptimization(std::shared_ptr<Graph>& graph) {
    FuseGraph(graph);
  }

  static bool needsGradient(const std::shared_ptr<const Graph>& graph) {
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

  void runTraced(Stack& stack) {
    const auto& state = tracer::getTracingState();
    auto inputs = last(stack, num_inputs);
    auto input_values = fmap(
        inputs, [](const IValue& v) { return tracer::getNestedValueTrace(v); });

    ArgumentSpec spec(
        autograd::GradMode::is_enabled(), inputs, num_flat_inputs);
    // NB: we could just run the fallback in here and call it a day, but that
    // would loose all the control flow information we have in the graph. Thus,
    // we run the fallback to get the correct output values, but we will
    // override the tracing states later.
    {
      // No need to trace a script module.
      ResourceGuard guard(tracer::pauseTracing());
      getOrCompileFallback().run(stack);
    }

    // Traces always have types propagated through them, so we make sure to
    // also propagate types through the graph we are inserting here.
    // However, this->graph itself may already have been generated with
    // tracing and so we only do the type propgation if no concrete types have
    // been set.
    auto local_graph = this->graph->copy();
    setInputTypes(*local_graph, spec);
    PropagateInputShapes(local_graph);
    auto output_values =
        inlineCallTo(*state->graph, *local_graph, input_values);

    auto outputs = last(stack, num_outputs);
    for (size_t i = 0; i < outputs.size(); ++i) {
      tracer::setValueTrace(outputs[i], output_values[i]);
    }
  }

  // The unoptimized starting graph. This field is effectively const, but we
  // can't make it so because Graph::copy() is not const (and making it const is
  // not that easy at this point).
  std::shared_ptr<Graph> graph;

  // If false, we'll run the graph as we get it, without any optimizations.
  // Useful for debugging.
  const bool optimize;
  const size_t num_inputs;
  const size_t num_flat_inputs; // Number of inputs, assuming all tuples would
                                // be flattened.
  const size_t num_outputs;

  // Populated only when optimize is false (and in that case plan_cache will be
  // unused). The compiled version of graph.
  ExecutionPlan fallback;

  // Mapping from argument configurations to optimized versions of the graph
  // that are specialized to the spec.
  std::unordered_map<ArgumentSpec, ExecutionPlan> plan_cache;

  // GraphExecutors can be accessed from multiple threads, so this thread needs
  // to be held every time we access the fallback or plan_cache.
  std::mutex compile_mutex;

  // Some tunable parameters
  size_t autodiffSubgraphNodeThreshold = 2;
  size_t autodiffSubgraphInlineThreshold = 5;
};

GraphExecutor::GraphExecutor(std::shared_ptr<Graph> graph, bool optimize)
    : pImpl(new GraphExecutorImpl(std::move(graph), optimize)) {}

void GraphExecutor::run(Stack& inputs) {
  return pImpl->run(inputs);
}

std::shared_ptr<Graph> GraphExecutor::graph() const {
  return pImpl->graph;
}

std::shared_ptr<Graph> GraphExecutor::graphFor(const Stack& inputs) const {
  return pImpl->graphFor(inputs);
}

GraphExecutorState GraphExecutor::getDebugState() {
  return pImpl->getDebugState();
}

void GraphExecutor::debugDisableAutodiffSubgraphInlining() {
  return pImpl->debugDisableAutodiffSubgraphInlining();
}

void runRequiredPasses(const std::shared_ptr<Graph>& g) {
  specializeUndef(*g);
  LowerGradOf(*g);
  // implicit inserted expand nodes are not necessarily always valid
  // when used inside script methods that might have unstable shapes
  // we remove the implicitly created ones, and have shape analysis
  // add valid expand nodes when the shapes are stable
  RemoveExpands(g);
  CanonicalizeOps(g);
  EliminateDeadCode(g);
}

} // namespace jit
} // namespace torch
