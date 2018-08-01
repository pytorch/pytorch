#include "torch/csrc/jit/graph_executor.h"

#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/jit/passes/batch_mm.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/erase_number_types.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/inplace_check.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/remove_expands.h"
#include "torch/csrc/jit/passes/decompose_addmm.h"
#include "torch/csrc/jit/passes/specialize_undef.h"
#include "torch/csrc/jit/passes/loop_unrolling.h"
#include "torch/csrc/jit/passes/lower_grad_of.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/ivalue.h"

#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/jit/script/compiler.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iterator>

namespace torch { namespace jit {

namespace {

using tensor_list = std::vector<at::Tensor>;
using Variable = autograd::Variable;
using autograd::variable_list;

// this type is in ExecutionPlan to run its Gradient if it is
// specified. It has a list of inputs captured by ExecutionPlan that
// it concats with inputs to form the full set of inputs to graph.
// see struct Gradient for a description of how the derivative graph
// is constructed and what variables are captured.
struct ExecutionPlanAutogradFunction : public autograd::Function {
  ExecutionPlanAutogradFunction(GraphExecutor graph, size_t capture_size)
  : graph(std::move(graph)) {
    is_var_capture.reserve(capture_size);
    var_captures.reserve(capture_size);
    ivalue_captures.reserve(capture_size);
  }

  virtual variable_list apply(variable_list&& inputs) override {
    Stack stack;
    stack.reserve(is_var_capture.size() + inputs.size());
    stack.insert(stack.end(), std::make_move_iterator(inputs.begin()),
                              std::make_move_iterator(inputs.end()));
    auto var_capture_it = var_captures.begin();
    auto ivalue_capture_it = ivalue_captures.begin();
    for (bool is_var : is_var_capture) {
      if (is_var) {
        stack.push_back(var_capture_it->unpack(this->shared_from_this()));
        ++var_capture_it;
      } else {
        stack.push_back(*ivalue_capture_it);
        ++ivalue_capture_it;
      }
    }
    graph.run(stack);
    return fmap(stack, [](IValue & val) {
      return autograd::Variable(std::move(val).toTensor());
    });
  }

  void capture(const IValue & val) {
    const bool is_tensor = val.isTensor();
    is_var_capture.push_back(is_tensor);
    if (is_tensor) {
      var_captures.emplace_back(Variable(val.toTensor()), false);
    } else {
      ivalue_captures.push_back(val);
    }
  }
private:
  friend struct ExecutionPlan;
  GraphExecutor graph;

  // INVARIANT: is_var_capture.size() == var_captures.size() + ivalue_captures.size()
  std::vector<bool> is_var_capture;
  std::vector<autograd::SavedVariable> var_captures;
  std::vector<IValue> ivalue_captures;
};

// an optimized way of executing the subgraph computed directly on
// tensors rather than Variables.
// This will unwrap Variables, run the plan, and re-wrap them.
// It can optionally also have a gradient which is hooked up
// to the output Variables if present.
struct ExecutionPlan {
  ExecutionPlan(std::shared_ptr<Graph>& graph)
      : f(graph),
        graph(graph),
        num_inputs(graph->inputs().size()),
        num_outputs(graph->outputs().size()) {}
  ExecutionPlan(std::shared_ptr<Graph>& graph, Gradient grad)
      : f(graph),
        graph(graph),
        grad(std::move(grad)),
        grad_executor(this->grad.df),
        num_inputs(graph->inputs().size()),
        num_outputs(graph->outputs().size()) {}

  void run(Stack & stack) const {
    if (grad) {
      return runWithGrad(stack);
    }
    InterpreterState(f).runOneStage(stack);
  }

  std::shared_ptr<Graph> get_graph() const {
    return graph;
  }

  ExecutionPlanState getDebugState() {
    ExecutionPlanState state;
    state.f = &f;
    state.graph = graph.get();
    if (grad) {
      state.grad = &grad;
      state.grad_executor = std::unique_ptr<GraphExecutorState>(
          new GraphExecutorState(grad_executor.getDebugState()));
    } else {
      state.grad = nullptr;
      state.grad_executor.reset();
    }
    return state;
  }

private:
  void detachVariables(Stack & stack) const {
    // It would be nice to use an ArrayRef here, but unfortunately those can only
    // return const references, so we need to do a bunch of indexing ourselves.
    const int64_t stack_size = stack.size();
    const int64_t stack_offset = stack_size - num_inputs;
    for (int64_t i = stack_offset; i < stack_size; ++i) {
      auto & v = stack[i];
      if (!v.isTensor()) continue;
      auto t = std::move(v).toTensor();
      v = IValue{t.defined() ? autograd::as_variable_ref(t).detach() : std::move(t)};
    }
  }
  // Capture (save) inputs that would be required to subsequently run backwards
  void captureInputs(ExecutionPlanAutogradFunction & grad_fn, at::ArrayRef<IValue> inputs) const {
    for (size_t offset : grad.df_input_captured_inputs) {
      grad_fn.capture(inputs[offset]);
    }
  }
  void captureOutputs(ExecutionPlanAutogradFunction & grad_fn, at::ArrayRef<IValue> outputs) const {
    for (size_t offset : grad.df_input_captured_outputs) {
      grad_fn.capture(outputs[offset]);
    }
  }

  // XXX: keep in mind that stack can be larger than the inputs we need!
  void runWithGrad(Stack & stack) const {
    auto grad_fn = std::make_shared<ExecutionPlanAutogradFunction>(grad_executor,
      grad.df_input_captured_inputs.size() + grad.df_input_captured_outputs.size());

    {
      auto inputs = last(stack, num_inputs);
      // hook up the outputs of df to the gradient functions of the inputs that require gradients
      for(auto idx : grad.df_output_vjps) {
        auto v = Variable(inputs[idx].toTensor());
        grad_fn->add_next_edge(v.gradient_edge());
      }
      captureInputs(*grad_fn, inputs);
    }

    detachVariables(stack);
    InterpreterState(f).runOneStage(stack);

    {
      auto outputs = last(stack, num_outputs);
      // hookup the gradients for the output tensors that require gradients
      // to the inputs to our gradient function df
      // TODO - XXX - if any output is the same tensor multiple times, views have to be
      // setup here. We need to refactor autograd until it is safe for
      // tensors to be constructed without all the viewing infrastructure.
      // this is currently intentionally not done here so we can get an idea of our
      // perf before introducing overhead for correctness
      for(auto idx : grad.df_input_vjps) {
        // Note: we have to set this up in place, or we have to throw away and
        // reallocate variables that were already created in wrapTensors. We
        // should add an API for this.
        Variable output = outputs[idx].toTensor();
        autograd::create_gradient_edge(output, grad_fn);
        output.set_requires_grad(true);
      }
      captureOutputs(*grad_fn, outputs);
      // drop the temporary outputs so that we return the same number of
      // outputs as if we were not also calculating gradient
      const size_t num_temporary_outputs = num_outputs - grad.f_real_outputs;
      stack.erase(stack.end() - num_temporary_outputs, stack.end());
    }
  }

  Code f;
  // optimized graph for debugging and testing
  std::shared_ptr<Graph> graph;
  // description of gradient as a graph
  Gradient grad; // if(grad) is false when this is unused
  // executor for df, including code caches
  GraphExecutor grad_executor;

  const size_t num_inputs;
  const size_t num_outputs;
};

} // anonymous namespace

// a Graph can be created via tracing, or via a language-based frontend
// GraphExecutor runs it. It can run the same graph on many different sizes
// and different requires_grad states, and handles specializations for each situation.
// GraphExecutor is completely unaware of tracing or module parameters to keep the
// tracing concerns separated.
struct GraphExecutorImpl {

  GraphExecutorImpl(std::shared_ptr<Graph> graph, bool optimize, bool symbolically_differentiable)
  : graph(std::move(graph))
  , optimize(optimize)
  , num_inputs(this->graph->inputs().size())
  , num_outputs(this->graph->outputs().size())
  , symbolically_differentiable(symbolically_differentiable)
  , may_introduce_gradient(calcMayIntroduceGradient(this->graph->block())) {}
  GraphExecutorImpl(std::shared_ptr<Graph> graph, bool optimize)
  : GraphExecutorImpl(graph, optimize, isDifferentiable(*graph)) {}

  // entry point where execution begins
  void run(Stack & stack) {
    if(stack.size() < num_inputs) {
      std::stringstream ss;
      ss << "expected " << num_inputs << " inputs but got " << stack.size() << " inputs";
      throw std::runtime_error(ss.str());
    }
    auto inputs = last(stack, num_inputs);

    // the tracer has called a graph executor
    // there is no need to optimize, but we do need to splice the graph of
    // this excutor into the trace. Otherwise we might unroll control-flow
    // operations.
    if(tracer::isTracing()) {
      return runTraced(stack);
    }

    // this is the fallback pathway, when we cannot differentiate
    if(!optimize || (!symbolically_differentiable && needsGradient(inputs))) {
      return runFallback(stack);
    }

    // either we can symbolically differentiate, or we do not need a gradient.
    // go down the route where we treat the inputs as tensors
    // and fully optimize
    auto & implementation = getOrCompile(inputs);
    return implementation.run(stack);
  }

  std::shared_ptr<Graph> graphFor(const Stack& stack) const {
    auto inputs = last(stack, num_inputs);
    ArgumentSpec spec(autograd::GradMode::is_enabled(), inputs);

    if (!optimize || (!symbolically_differentiable && needsGradient(inputs))) {
      JIT_ASSERTM(autograd_fallback_graph, "No graph found for given inputs");
      return autograd_fallback_graph;
    }

    auto it = plan_cache.find(spec);
    JIT_ASSERTM(it != plan_cache.end(), "No graph found for given inputs");
    return it->second.get_graph();
  }

  GraphExecutorState getDebugState() {
    GraphExecutorState state;
    state.graph = graph.get();
    if (autograd_fallback) {
      state.autograd_fallback = &autograd_fallback;
      state.autograd_fallback_graph = autograd_fallback_graph.get();
    } else {
      state.autograd_fallback = nullptr;
      state.autograd_fallback_graph = nullptr;
    }
    for (auto & entry : plan_cache) {
      state.execution_plans.emplace(entry.first, entry.second.getDebugState());
    }
    return state;
  }

private:
  friend struct GraphExecutor;

  void runTraced(Stack & stack) {
    auto state = tracer::getTracingState();
    auto inputs = last(stack, num_inputs);
    auto input_values = fmap(inputs, [](const IValue & v) {
      return tracer::getValueTrace(v.toTensor());
    });

    ArgumentSpec spec(autograd::GradMode::is_enabled(), inputs);
    runFallback(stack);

    auto all_dynamic = [](const at::ArrayRef<Value*> xs) {
      for(Value* x : xs) {
        if(x->type()->kind() != TypeKind::DynamicType)
          return false;
      }
      return true;
    };
    // Traces always have types propagated through them, so we make sure to
    // also propagate types through the graph we are inserting here.
    // However, this->graph itself may already have been generated with
    // tracing and so we only do the type propgation if no concrete types have
    // been set.
    auto local_graph = this->graph;
    if(all_dynamic(local_graph->inputs()) && all_dynamic(local_graph->outputs())) {
      local_graph = this->graph->copy();
      PropagateInputShapes(*local_graph, spec);
    }
    auto output_values = script::inlineCallTo(*state->graph, *local_graph, input_values);

    auto outputs = last(stack, num_outputs);
    for (size_t i = 0; i < outputs.size(); ++i) {
      // We can't attach tracing states to scalars, so we have to skip them here
      // TODO: Should we reinterpret them as scalar tensors instead?
      if (!outputs[i].isTensor()) continue;
      tracer::setValueTrace(outputs[i].toTensor(), output_values[i]);
    }
  }

  void runFallback(Stack & stack) {
    auto & fb = getOrCreateAutogradFallback();
    InterpreterState(fb).runOneStage(stack);
  }

  static bool calcMayIntroduceGradient(Block* b) {
    for(Node* n : b->nodes()) {
      if(n->kind() == prim::PythonOp)
        return true;
      for(Block* bb : n->blocks()) {
        if(calcMayIntroduceGradient(bb))
          return true;
      }
    }
    return false;
  }
  bool needsGradient(at::ArrayRef<IValue> inputs) const {
    if (!autograd::GradMode::is_enabled()) {
      return false;
    }
    if (may_introduce_gradient)
      return true;
    for (const IValue & value : inputs) {
      if (!value.isTensor()) continue;
      auto t = value.toTensor();
      if (t.defined() && autograd::as_variable_ref(t).requires_grad())
        return true;
    }
    return false;
  }

  const Code & getOrCreateAutogradFallback() {
    std::lock_guard<std::mutex> lock(compile_mutex);
    if(autograd_fallback) {
      return autograd_fallback;
    }
    auto graph_ = graph->copy();
    runRequiredPasses(graph_);
    if(optimize) {
      if(!symbolically_differentiable) {
        EraseShapeInformation(*graph_);
        CreateAutodiffSubgraphs(*graph_);
      }
      runOptimization(graph_, /*graphMustSupportVariables=*/true);
    }
    autograd_fallback_graph = graph_;
    autograd_fallback = Code(graph_);
    return autograd_fallback;
  }
  const ExecutionPlan & getOrCompile(at::ArrayRef<IValue> inputs) {
    // outside lock guard, to minimize the time holding the lock on the fast path
    // ArgumentSpec even computes its hashCode here.
    ArgumentSpec spec(autograd::GradMode::is_enabled(), inputs);
    {
      std::lock_guard<std::mutex> lock(compile_mutex);
      auto it = plan_cache.find(spec);
      if(it != plan_cache.end())
        return it->second;
      auto plan = compileSpec(spec);
      auto r = plan_cache.emplace(std::move(spec), std::move(plan));
      return r.first->second;
    }
  }

  bool argumentSpecRequiresGradient(const ArgumentSpec & spec) {
    for(size_t i = 0; i < spec.size(); ++i) {
      if(spec.at(i).requires_grad())
        return true;
    }
    return false;
  }

  ExecutionPlan compileSpec(const ArgumentSpec & spec) {
    auto graph_ = graph->copy();

    specializeToSpec(graph_, spec);

    if(!argumentSpecRequiresGradient(spec)) {
      runOptimization(graph_, /*graphMustSupportVariables=*/false);
      return ExecutionPlan(graph_);
    }
    JIT_ASSERT(symbolically_differentiable);

    std::vector<bool> requires_grads;
    requires_grads.reserve(spec.size());
    for(size_t i = 0; i < spec.size(); i++)
      requires_grads.push_back(spec.at(i).requires_grad());

    Gradient gradient = differentiate(graph_, requires_grads);
    graph_ = gradient.f;
    runOptimization(graph_, /*graphMustSupportVariables=*/false);
    return ExecutionPlan(graph_, std::move(gradient));
  }
  // the unoptimized starting graph
  // this is never mutated
  std::shared_ptr<Graph> graph;

  // true - do everything we can to make this graph run fast
  // false - do not modifiy the graph at all and just use the interpreter
  // to run the graph. Useful for debugging correctness issues in the implementation
  const bool optimize;
  const size_t num_inputs;
  const size_t num_outputs;

  // GraphExecutor optimizes more aggresively when we _know_ the graph will be
  // symbolically differentiable.
  bool symbolically_differentiable;

  // some ops, including python operations, can intorduce requires_grad=True
  // variables even though no inputs to this graph are availiable, if
  // the graph includes those operators then needGradient must be true
  // regardles of input state.
  bool may_introduce_gradient;

  // when this graph has some parts that are not symbolically_differentable,
  // but some input does require a derivative, we create and use autograd_fallback,
  // which wraps up the fully differentiable subgraphs, and then runs the outer
  // graph through autograd.
  // Since we can't optimize black box functions anyway, there is only one fallback path,
  // and it must work on all sizes (so no optimizations that inspect sizes can run on it)
  std::shared_ptr<Graph> autograd_fallback_graph;
  Code autograd_fallback;

  // optimizable code paths, used when we can differentiate or when no derivative is needed
  // Spec describes input conditions, Plan describes how to execute them.
  std::unordered_map<ArgumentSpec, ExecutionPlan> plan_cache;

  // GraphExecutor can be accessed from  multiple thread so
  // anytime we are checking or updating the autograd_fallback or
  // plan_cache, we must hold the compile mutex.
  // along the fast path (no compilation) code should
  // hold this for as little time as possible.
  std::mutex compile_mutex;
};

GraphExecutor::GraphExecutor(std::shared_ptr<Graph> graph, bool optimize)
: pImpl(new GraphExecutorImpl(std::move(graph), optimize)) {}

GraphExecutor::GraphExecutor(std::shared_ptr<Graph> graph, bool optimize, bool symbolically_differentiable)
: pImpl(new GraphExecutorImpl(std::move(graph), optimize, symbolically_differentiable)) {}

void GraphExecutor::run(Stack & inputs) {
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


void runRequiredPasses(const std::shared_ptr<Graph>& g)  {
  LowerGradOf(*g);
  // implicit inserted expand nodes are not necessarily always valid
  // when used inside script methods that might have unstable shapes
  // we remove the implicitly created ones, and have shape analysis
  // add valid expand nodes when the shapes are stable
  RemoveExpands(g);
}

void specializeToSpec(const std::shared_ptr<Graph>& graph, const ArgumentSpec& spec) {
  // clean up GradOf and AutogradAdd nodes
  // this must be first because later passes do not know what GradOfs are
  std::vector<bool> defined;
  for(size_t i = 0; i < spec.size(); ++i) {
    defined.push_back(spec.at(i).defined());
  }
  specializeUndef(*graph, defined);

  // required passes shared with autograd fallback
  runRequiredPasses(graph);

  // Decompose addmm nodes to add + mm, so expands can be inserted and
  // gradients accumulated on the backward pass
  //
  // In the future, if we need more passes like this, we should convert this
  // into a generic canonicalization pass.
  DecomposeAddmm(graph);
  // clean up dead constants from specialization
  EliminateDeadCode(graph);
  // calculate all input shapes
  PropagateInputShapes(*graph, spec);
}

void runOptimization(std::shared_ptr<Graph> & graph, bool graphMustSupportVariables) {

  // these optimizations must run in the presence of variables
  // and when shape information is not statically known.
  EliminateDeadCode(graph);
  CheckInplace(graph);
  EliminateCommonSubexpression(graph);

  if (!graphMustSupportVariables) {
    // These optimizations can introduce operators like FusionGroup that
    // do not work on variables

    // They also may assume that concrete sizes/strides are availiable
    UnrollLoops(graph);
    ConstantPropagation(graph);
    //TODO: create peephole optimizations that are safe to run
    // when we are using variables, and when we do not know sizes.
    PeepholeOptimize(graph);
    // TODO: remove mandatory size checking in BatchMM, otherwise
    // it works fine on variables.
    BatchMM(graph);
    FuseGraph(graph);
  }
}

}}
