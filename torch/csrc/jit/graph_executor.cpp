#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/autograd/function.h"

#include <unordered_map>

namespace torch { namespace jit {

// input captures, output captures, temporary captures

/*
// how to run a single optimized execution
// optimized executions unwrap/re-wrap tensors as described in the Gradient struct.
class Plan {
  Code code;
  Gradient? gradient;
  variable_list run(variable_list inputs) {
    tensors = unwrap_variables(inputs);
    outputs = code.run(tensors);
    output_vars = wrap_in_nograd_variables(outputs)
    if(!gradient) {
      return output_vars;
    }
    grad_fn = GradientFunction(gradient);
    for(x : gradient.gradient_inputs) {
      output_vars[x].requires_grad = true
      output_vars[x].grad_fn = grad_fn;
      output_vars[x].output_nr = grad_fn.num_inputs++;
    }
    for(x : gradient.outputs) {
      grad_fn.next_functions.push_back(inputs[x].grad_fn, inputs[x].output_nr)
      // NOTE: it would be great if edge_type was struct InputPort { Gradient grad_fn; int offset; }
      // and Variable also had an InputPort instead of grad_fn+output_nr.
      // then this would just be:
      // grad_fn.next_functions.push_back(inputs[x].input_port);
    }
    // capture needs to be separate from the grad_fn constructor since
    // it captures some of the output vars whose gradients point to grad_fn itself.
    grad_fn.capture(inputs, outputs_vars);
    // remove the last gradient.num_temporary_outputs from output_vars, they are not used outside
    return output_vars.shorten_by(gradient.num_temporary_outputs);
    // note: those temporary vars were capture by the gradient, so if we take _its_ gradient
    // it is possible they are used.
  }
};


class GradientFunction : autograd::Function {
  GradientFunction(Gradient grad)
  : grad(grad) {}
  variable_list run(variable_list inputs) {
    return grad.graph_exe.run(captures+inputs); //concern 2: costly concat, captures can be really big
  }
  void capture(variable_list inputs, variable_list outputs) {
    for(c : grad.captures) {
      captures.push_back(c.kind == Input ? inputs[c.index] : outputs[c.index]);
    }
  }
  Gradient grad;
  saved_variable_list captures; //what is saved variable in c++
};

*/

using tensor_list = std::vector<at::Tensor>;
using variable_tensor_list = tensor_list;

struct ExecutionPlan;
struct GraphExecutor;

struct ExecutionPlanAutogradFunction : public autograd::Function {
  ExecutionPlanAutogradFunction(std::shared_ptr<GraphExecutor> graph)
  : graph(std::move(graph)) {}
  virtual autograd::variable_list apply(const autograd::variable_list& inputs) override;
private:
  friend struct ExecutionPlan;
  std::shared_ptr<GraphExecutor> graph;
  std::vector<autograd::SavedVariable> captures;
};


struct ExecutionPlan {
  ExecutionPlan(std::shared_ptr<Graph> & graph)
  : f(graph) {}
  ExecutionPlan(std::shared_ptr<Graph> & graph, Gradient grad)
  : f(graph), grad(std::move(grad)) {}

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
    auto grad_fn = std::make_shared<ExecutionPlanAutogradFunction>(grad_executor, 0);
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
  friend struct ExecutionPlanAutogradFunction;
  Code f;

  // description of gradient as a graph
  Gradient grad; // if(grad) is false when this is unused
  // executor for df, including code caches
  std::shared_ptr<GraphExecutor> grad_executor;
};

// a Graph can be created via tracing, or via a language-based frontend
// GraphExecutor runs it. It can run the same graph on many different sizes
// and different requires_grad states, and handles specializations for each situation.
// GraphExecutor is completely unaware of tracing or module parameters to keep the
// tracing concerns separated.
struct GraphExecutor {

  // we dynamically expect this list of at::Tensor to be Variables.
  // TODO: make these arguments safe.
  using variable_tensor_list = tensor_list;
  using Variable = autograd::Variable;
  GraphExecutor(std::shared_ptr<Graph> graph, bool symbolically_differentiable)
  : graph(std::move(graph))
  , symbolically_differentiable(symbolically_differentiable) {}
  GraphExecutor(std::shared_ptr<Graph> graph)
  : symbolically_differentiable(isDifferentiable(*graph)) {}

  // entry point where execution begins
  variable_tensor_list run(variable_tensor_list inputs) {

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
      auto new_node = new_g->createClone(node, [&](Value* v) {
        return value_map.at(v);
      });
      for(size_t i = 0; i < node->outputs().size(); ++i) {
        value_map[node->outputs()[i]] = new_node->outputs()[i];
        new_node->outputs()[i]->copyMetadata(node->outputs()[i]);
      }
    }
    return new_g;
  }
  void optimize(Graph & graph, bool graphMustSupportVariables) {
    //TODO
  }
  Code & getOrCreateAutogradFallback() {
    std::lock_guard<std::mutex> lock(compile_mutex);
    if(autograd_fallback) {
      return autograd_fallback;
    }
    auto graph_ = copy(*graph);
    CreateAutodiffSubgraphs(*graph_);
    optimize(*graph_, /*graphMustSupportVariables=*/true);
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
  void SpecializeToSpec(Graph & g, const ArgumentSpec & spec) {
    PropagateInputShapes(g, spec);
    for(size_t i = 0; i < spec.size(); ++i) {
      if(!spec.tensorInfo(i).defined()) {
        
      }
    }
  }
  ExecutionPlan compileSpec(const ArgumentSpec & spec) {
    auto graph_ = copy(*graph);
    SpecializeToSpec(*graph_, spec);
    if(!gradientIsPossible(spec)) {
      optimize(*graph_, /*graphMustSupportVariables=*/false);
      return ExecutionPlan(graph_);
    }
    JIT_ASSERT(symbolically_differentiable);
    // TODO: add in requires_grad flags when PR is merged
    Gradient gradient = differentiate(graph_);
    graph_ = gradient.f;
    optimize(*graph_, /*graphMustSupportVariables=*/false);
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

autograd::variable_list
ExecutionPlanAutogradFunction::apply(const autograd::variable_list& inputs) {
  // TODO: stupid copies here to convert to/from tensor_list
  auto tensors = graph->run(variable_tensor_list(inputs.begin(), inputs.end()));
  return autograd::variable_list(tensors.begin(), tensors.end());
}

}}

/*

class GraphExecutor {


  void optimize(Graph graph, bool graphMustSupportVariables) {
    propagate_undef(graph); // remove all compute related to known undefined nodes, needed because a lot of temporaries on single-backward
                            // will have undefined grads since the variables were never used
    <standard optimization passes>
    if(!graphMustSupportVariables) {
      // some optimizations, like fusion groups, are not valid if we need to run
      // variables through the graph. Others like rewriting tensor ops into
      // simpler tensor ops, are safe.
      // once we know that something is not going to be used to run variables, then
      // we can perform the unsafe ones.
      perform_fusion(graph);
    }
  }

};






*/
