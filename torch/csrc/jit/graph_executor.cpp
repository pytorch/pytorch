#include "Python.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/inplace_check.h"
#include "torch/csrc/jit/passes/batch_mm.h"

#include "torch/csrc/autograd/function.h"

#include <unordered_map>

namespace torch { namespace jit {

namespace {

using tensor_list = std::vector<at::Tensor>;
using variable_tensor_list = tensor_list;
using Variable = autograd::Variable;
using autograd::variable_list;

struct ExecutionPlanAutogradFunction : public autograd::Function {
  ExecutionPlanAutogradFunction(GraphExecutor graph)
  : graph(std::move(graph)) {}
  virtual variable_list apply(const variable_list& inputs) override {
    // TODO: stupid copies here to convert to/from tensor_list
    // TODO: things are not moved correctly
    variable_tensor_list all_inputs;
    all_inputs.reserve(captures.size() + inputs.size());
    for(auto & sv : captures) {
      all_inputs.push_back(sv.unpack(this->shared_from_this()));
    }
    all_inputs.insert(all_inputs.end(), inputs.begin(), inputs.end());
    auto tensors = graph.run(std::move(all_inputs));
    return autograd::variable_list(tensors.begin(), tensors.end());
  }
private:
  friend struct ExecutionPlan;
  GraphExecutor graph;
  std::vector<autograd::SavedVariable> captures;
};


struct ExecutionPlan {
  ExecutionPlan(std::shared_ptr<Graph> & graph)
  : f(graph) {
    std::cout << "creating plan for: " << *graph << "\n";
  }
  ExecutionPlan(std::shared_ptr<Graph> & graph, Gradient grad)
  : f(graph), grad(std::move(grad)), grad_executor(grad.df) {}

  variable_tensor_list run(variable_tensor_list inputs) {
    if(grad) {
      return runWithGrad(std::move(inputs));
    }
    unwrapVariables(inputs);
    tensor_list outputs;
    InterpreterState(f).runOneStage(std::move(inputs), outputs);
    wrapTensors(outputs);
    return outputs;
  }
private:
  void unwrapVariables(tensor_list & list) {
    for(auto & v : list) {
      v = autograd::Variable(v).data();
    }
  }
  void wrapTensors(tensor_list & list) {
    for(auto & v : list) {
      v = autograd::make_variable(v);
    }
  }
  void captureInputs(ExecutionPlanAutogradFunction & grad_fn, variable_tensor_list & inputs) {
    auto & capture_desc = grad.df_input_captures;
    size_t N = capture_desc.size();
    for(size_t i = 0; i < N; ++i) {
      if(capture_desc[i].kind == Capture::Kind::Input) {
        size_t offset = capture_desc[i].offset;
        grad_fn.captures[i] = autograd::SavedVariable(autograd::Variable(inputs[offset]), false);
      }
    }
  }
  void captureOutputs(ExecutionPlanAutogradFunction & grad_fn, variable_tensor_list & outputs) {
    auto & capture_desc = grad.df_input_captures;
    size_t N = capture_desc.size();
    for(size_t i = 0; i < N; ++i) {
      if(capture_desc[i].kind == Capture::Kind::Output) {
        size_t offset = capture_desc[i].offset;
        grad_fn.captures[i] = autograd::SavedVariable(autograd::Variable(outputs[offset]), true);
      }
    }
  }

  variable_tensor_list runWithGrad(variable_tensor_list inputs) {
    auto grad_fn = std::make_shared<ExecutionPlanAutogradFunction>(grad_executor);
    // hook up the outputs of df to the gradient functions of the inputs that require
    // gradients
    for(auto idx : grad.df_output_vjps) {
      autograd::Variable v(inputs[idx]);
      // TODO: this kinda stuff is _way_ to low level to the public API of variable.
      // Why do I have to care here whether v has a grad_fn or grad accumulator?
      // Why do I have to care here about output_nr? I just want to say
      // grad_fn->setOutputTo(i, v.input_port());
      grad_fn->next_functions.emplace_back(v.grad_fn() ? v.grad_fn() : v.grad_accumulator(), v.output_nr());
    }
    captureInputs(*grad_fn, inputs);

    unwrapVariables(inputs);
    tensor_list outputs;
    InterpreterState(f).runOneStage(std::move(inputs), outputs);
    wrapTensors(outputs);
    // hookup the gradients for the output tensors that require gradients
    // to the inputs to our gradient function df
    // XXX - if any output is the same tensor multiple times, views have to be
    // setup here. We need to refactor autograd until it is safe for
    // tensors to be constructed without all the viewing infrastructure.
    for(auto idx : grad.df_input_vjps) {
      autograd::Variable o(outputs[idx]);
      auto impl = o.get();
      // Note: we have to set this up in place, or we have to
      // throw away and reallocate variables that were already created in
      // wrapTensors. We should add an API for this.
      impl->_grad_fn = grad_fn;
      impl->output_nr = grad_fn->num_inputs++;
      impl->_requires_grad = true;
    }
    captureOutputs(*grad_fn, outputs);
    // drop the temporary outputs so that we return the same number of
    // outputs as if we were not also calculating gradient
    outputs.erase(outputs.begin() + grad.f_real_outputs, outputs.end());
    return outputs;
  }
  Code f;
  // description of gradient as a graph
  Gradient grad; // if(grad) is false when this is unused
  // executor for df, including code caches
  GraphExecutor grad_executor;
};

} // anonymous namespace

// a Graph can be created via tracing, or via a language-based frontend
// GraphExecutor runs it. It can run the same graph on many different sizes
// and different requires_grad states, and handles specializations for each situation.
// GraphExecutor is completely unaware of tracing or module parameters to keep the
// tracing concerns separated.
struct GraphExecutorImpl {

  // we dynamically expect this list of at::Tensor to be Variables.
  // TODO: make these arguments safe.
  GraphExecutorImpl(std::shared_ptr<Graph> graph, bool symbolically_differentiable)
  : graph(std::move(graph))
  , symbolically_differentiable(symbolically_differentiable) {}
  GraphExecutorImpl(std::shared_ptr<Graph> graph)
  : graph(std::move(graph))
  , symbolically_differentiable(isDifferentiable(*this->graph)) {}

  // entry point where execution begins
  variable_tensor_list run(variable_tensor_list inputs) {
    std::cout << "GE Run\n";
    // this is the fallback pathway
    if(!symbolically_differentiable && gradientIsPossible(inputs)) {
      auto & fb = getOrCreateAutogradFallback();
      InterpreterState state(fb);
      tensor_list outputs;
      state.runOneStage(std::move(inputs), outputs);
      // note: we never unwrapped inputs, because we want autograd to record the trace
      return outputs;
    }

    // either we can symbolically differentiate, or we do not need a gradient.
    // go down the route where we treat the inputs as tensors
    // and fully optimize
    auto & implementation = getOrCompile(inputs);
    return implementation.run(std::move(inputs));
  }

private:

  static bool gradientIsPossible(const variable_tensor_list & inputs) {
    if (!autograd::GradMode::is_enabled()) {
      return false;
    }
    for (const auto & tensor : inputs) {
      if(tensor.defined() && Variable(tensor).requires_grad())
        return true;
    }
    return false;
  }
  static bool isDifferentiable(Graph & g) {
    for(auto n : g.nodes()) {
      if(!jit::isDifferentiable(n))
        return false;
    }
    return true;
  }
  //TODO: move somewhere reasonable
  static std::shared_ptr<Graph> copy(Graph & g) {
    auto new_g = std::make_shared<Graph>();
    std::unordered_map<Value*, Value*> value_map;
    for(auto input : g.inputs()) {
      value_map[input] = new_g->addInput()->copyMetadata(input);
    }
    for(auto node : g.nodes()) {
      auto new_node = new_g->appendNode(new_g->createClone(node, [&](Value* v) {
        return value_map.at(v);
      }));
      for(size_t i = 0; i < node->outputs().size(); ++i) {
        value_map[node->outputs()[i]] = new_node->outputs()[i];
        new_node->outputs()[i]->copyMetadata(node->outputs()[i]);
      }
    }
    for(auto output : g.outputs()) {
      new_g->registerOutput(value_map.at(output));
    }
    return new_g;
  }
  void optimize(std::shared_ptr<Graph> & graph, bool graphMustSupportVariables) {
    // Now, we have a complete trace. Compile it.
    EliminateDeadCode(graph);
    CheckInplace(graph);
    EliminateCommonSubexpression(graph);
    if (!graphMustSupportVariables) {
      PeepholeOptimize(graph);
      BatchMM(graph);
      FuseGraph(graph);
    }
  }
  Code & getOrCreateAutogradFallback() {
    std::lock_guard<std::mutex> lock(compile_mutex);
    if(autograd_fallback) {
      return autograd_fallback;
    }
    auto graph_ = copy(*graph);
    CreateAutodiffSubgraphs(*graph_);
    optimize(graph_, /*graphMustSupportVariables=*/true);
    autograd_fallback = Code(graph_);
    return autograd_fallback;
  }
  ExecutionPlan & getOrCompile(const variable_tensor_list & inputs) {
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
  bool gradientIsPossible(const ArgumentSpec & spec) {
    for(size_t i = 0; i < spec.size(); ++i) {
      if(spec.tensorInfo(i).requires_grad())
        return true;
    }
    return false;
  }


  // remove ReplaceIfUndef(v, replacement) nodes that consume inputs with 'v' if
  // the input is defined, and 'replacement' if it is not.
  void specializeUndef(Graph & g, const ArgumentSpec & spec) {
    for(size_t i = 0; i < spec.size(); i++) {
      std::vector<Value*> to_replace;
      // do not edit in place, since it invalides uses iterator
      for(auto u : g.inputs()[i]->uses()) {
        if(u.user->kind() == kReplaceIfUndef) {
          to_replace.push_back(u.user->output());
        }
      }
      for(auto v : to_replace) {
        int idx = spec.tensorInfo(i).defined() ? 0 : 1;
        v->replaceAllUsesWith(v->node()->inputs()[idx]);
        v->node()->destroy();
      }
    }
  }
  bool isZero(Node * n) {
    return n->kind() == kConstant &&
      n->hasAttribute(kis_zero) &&
      n->i(kis_zero);
  }
  // a + 0 -> a
  // 0 + a -> a
  void propagateZeros(Graph & g) {
    for(auto it = g.nodes().begin(); it != g.nodes().end(); ++it) {
      if(it->kind() == kadd && at::Scalar(it->t(kalpha)).toDouble() == 1.0) {
        if(isZero(it->inputs()[0]->node())) {
          it->output()->replaceAllUsesWith(it->inputs()[1]);
          it.destroyCurrent();
        } else if(isZero(it->inputs()[1]->node())) {
          it->output()->replaceAllUsesWith(it->inputs()[0]);
          it.destroyCurrent();
        }
      }
    }
  }
  void SpecializeToSpec(Graph & g, const ArgumentSpec & spec) {
    // clean up replaceIfUndef nodes
    specializeUndef(g, spec);
    // clean up additions resulting from nodes that were in fact undefined
    propagateZeros(g);

    // calculate all input shapes
    PropagateInputShapes(g, spec);
  }
  ExecutionPlan compileSpec(const ArgumentSpec & spec) {
    std::cout << "graph: " << *graph << "\n";
    auto graph_ = copy(*graph);
    std::cout << "graph: " << *graph_ << "\n";
    SpecializeToSpec(*graph_, spec);
    std::cout << "graph: " << *graph_ << "\n";
    if(!gradientIsPossible(spec)) {
      optimize(graph_, /*graphMustSupportVariables=*/false);
      std::cout << "graph: " << *graph_ << "\n";
      return ExecutionPlan(graph_);
    }
    JIT_ASSERT(symbolically_differentiable);
    // TODO: add in requires_grad flags when PR is merged
    std::vector<bool> requires_grads;
    for(size_t i = 0; i < spec.size(); i++)
      requires_grads.push_back(spec.tensorInfo(i).requires_grad());
    Gradient gradient = differentiate(graph_, requires_grads);
    graph_ = gradient.f;
    optimize(graph_, /*graphMustSupportVariables=*/false);
    return ExecutionPlan(graph_, std::move(gradient));
  }
  // the unoptimized starting graph
  // this is never mutated
  std::shared_ptr<Graph> graph;
  // GraphExecutor optimizes more aggresively when we _know_ the graph will be
  // symbolically differentiable.
  bool symbolically_differentiable;
  // optimizable code paths, used when we can differentiate or when no derivative is needed

  // when this graph has some parts that are not symbolically_differentable,
  // but some input does require a derivative, we create and use autograd_fallback,
  // which wraps up the fully differentiable subgraphs, and then runs the outer
  // graph through autograd.
  // Since we can't optimize black box functions anyway, there is only one fallback path,
  // and it must work on all sizes (so no optimizations that inspect sizes can run on it)
  Code autograd_fallback;

  // Spec describes input conditions, Plan describes how to execute them.
  std::unordered_map<ArgumentSpec, ExecutionPlan> plan_cache;

  // GraphExecutor can be accessed from  multiple thread so
  // anytime we are checking or updating the autograd_fallback or
  // plan_cache, we must hold the compile mutex
  // along the fast path (no compilation) code should
  // hold this for as little time as possible.
  std::mutex compile_mutex;
};

GraphExecutor::GraphExecutor(std::shared_ptr<Graph> graph)
: pImpl(new GraphExecutorImpl(std::move(graph))) {}

GraphExecutor::GraphExecutor(std::shared_ptr<Graph> graph, bool symbolically_differentiable)
: pImpl(new GraphExecutorImpl(std::move(graph), symbolically_differentiable)) {}

std::vector<at::Tensor> GraphExecutor::run(std::vector<at::Tensor> inputs) {
  return pImpl->run(std::move(inputs));
}

}}
