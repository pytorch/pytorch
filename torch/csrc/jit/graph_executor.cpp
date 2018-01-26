#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/shape_analysis.h"

#include <unordered_map>

namespace torch { namespace jit {


struct Gradient {
};
struct ExecutionPlan {
  using variable_list = autograd::variable_list;
  variable_list run(variable_list inputs) {
    //TODO
  }
  ExecutionPlan(std::shared_ptr<Graph> & graph, std::unique_ptr<Gradient> grad)
  : f(graph), grad(std::move(grad)) {}
private:
  Code f;
  std::unique_ptr<Gradient> grad;
};

// a Graph can be created via tracing, or via a language-based frontend
// GraphExecutor runs it. It can run the same graph on many different sizes
// and different requires_grad states, and handles specializations for each situation.
// GraphExecutor is completely unaware of tracing or module parameters to keep the
// tracing concerns separated.
struct GraphExecutor {
  using variable_list = autograd::variable_list;
  using tensor_list = std::vector<at::Tensor>;
  GraphExecutor(std::shared_ptr<Graph> graph, bool symbolically_differentiable)
  : graph(std::move(graph))
  , symbolically_differentiable(symbolically_differentiable) {}
  GraphExecutor(std::shared_ptr<Graph> graph)
  : symbolically_differentiable(isDifferentiable(*graph)) {}

  // entry point where execution begins
  variable_list run(variable_list inputs) {

    // this is the fallback pathway
    if(!symbolically_differentiable && gradientIsPossible(inputs)) {
      auto & fb = getOrCreateAutogradFallback();
      InterpreterState state(fb);
      tensor_list outputs;
      state.runOneStage(toTensorList(std::move(inputs)), outputs);
      // note: we never unwrapped inputs, because we want autograd to record the trace
      return toVariableList(std::move(outputs));
    }

    // either we can symbolically differentiate, or we do not need a gradient.
    // go down the route where we treat the inputs as tensors
    // and fully optimize
    auto & implementation = getOrCompile(inputs);
    return implementation.run(std::move(inputs));
  }

private:
  // TODO: sort out variable_list tensor_list bs
  static tensor_list toTensorList(variable_list inputs) {
    return tensor_list(inputs.begin(), inputs.end());
  }
  static variable_list toVariableList(tensor_list inputs) {
    return variable_list(inputs.begin(), inputs.end());
  }

  static bool gradientIsPossible(const variable_list & inputs) {
    if (!autograd::GradMode::is_enabled()) {
      return false;
    }
    for (const auto & tensor : inputs) {
      if(tensor.defined() && tensor.requires_grad())
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
  ExecutionPlan & getOrCompile(const variable_list & inputs) {
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
  ExecutionPlan compileSpec(const ArgumentSpec & spec) {
    auto graph_ = copy(*graph);
    PropagateInputShapes(*graph_, spec);
    if(!gradientIsPossible(spec)) {
      optimize(*graph_, /*graphMustSupportVariables=*/false);
      return ExecutionPlan(graph_, nullptr);
    }
    JIT_ASSERT(symbolically_differentiable);
    std::unique_ptr<Gradient> gradient = nullptr;
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

}}
/*

class GraphExecutor {

  Plan compileSpec(spec) {
    graph_ = copy(graph); // make a copy for the specialization
    specialize_to(spec);
    if(!spec.gradientIsPossible()) {
      // we do not need to compute a gradient
      optimize(graph_, spec, graphMustSupportVaribales=false)
      return Plan(Code(graph), gradient=None)
    }
    assert(symbolically_differentiable);
    gradient = get_symblic_gradient(graph_);
    optimize(graph_, graphMustSupportVariables=false)
    return Plan(Code(graph_), gradient)
  }


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


  // PR exists
  // propagate the types and shapes describe in spec across the entire graph
  void type_shape_propagate(graph, spec) {
    // <fake it by making up fake tensors, running the op, and throwing away the result>
    // incrementally add real shape/type propagation by implementing the expensive
    // ops (matmult, conv, etc.) first
  }

  // make a graph specific to the sizes/strides/defined-ness, etc. in spec
  void specialze_to(Graph graph, Spec spec) {
    type_shape_propagate(graph, spec);
  }

  // PR exists
  Gradient get_symbolic_gradient(graph_) {
    // original graph y0,y1,m,y2 = f(x0,x1,n,x2) // assume m and n are requires_grad=False

    // modified original graph y0,y1,m,y2,t0,t1 = f(x0,x1,n,x2)
    // t0, t1, ... are all the temporaries needed to compute the gradient
    // gradient graph may require some of the inputs of f as well.
    // gradient graph  [dx0, dx1, dx2] = f'(x0,x2,t0,t1,dy0, dy1, dy2, dt0, dt1)
    // note: dt0 and dt1 will be undefined _unless_ there is a double backward, in which case,
    // since f' uses t0 and t1, the double backward will produce gradients for them

    // implementation in two stages:
    // stage1: insert_symbolic_gradient(graph_), which works in the original graph adding the gradient expressions
    // stage2: split_off_stage1_graph which will figure out what is required as temporaries, creating new outputs for them if needed.
    // and creating a gradient_graph object.
    return Gradient(gradient_graph, <metadata as described above about how to hook up the gradient in Gradient object>);
  }

};


// Definition of Spec
// CONCERN 1: the key is huuuuuuge: with some graphs having the entire
// set of parameters as inputs each time!
// we have similar keys today, but it still worries me.
// mitigation: we only use specs for symbolically differentiable ops, so maybe in practice what is checked is smaller
// PR Exists
class Spec {
  TensorInfo tensors;
  bool gradientIsPossible() {
    <any TensorInfo has requires_grad>
  }
};

// PR Exists
class TensorInfo {
  ScalarType type;
  int device;
  list[int] sizes;
  list[int] strides; //fusion groups needs this
  bool defined;
  bool requires_grad; // true if its a variable, and nograd is not set on this execution, and the variable itself requires grad
};

class Code; // already implemented, this is the 'bytecode' that the interpreter can run

// holds everything needed to define the gradient when Plan is run

// This is a purely symbolic representation of the Gradient
// (it has a GraphExecutor because we need somewhere to put the
// code cache for the executor.)  Morally, Gradient has a Graph
// (and we cache the GraphExecutor somewhere.)
class Gradient {
  GraphExecutor gradient_exe; // definition of f', the gradient of f

  // meta-data describing how the gradient function is hooked up in the autograd

  // INPUTS
  // one for each input to f', these are
  // descriptions of where the inputs of f' come from, the first set of inputs
  // are always captured from the inputs/outputs of f  (temporaries in f get promoted to outputs if needed)
  list<Capture> captured_inputs;
  // the remaining inputs are inputs to the grad_fn object
  // each integer specifies an output of f whose gradient (grad_fn, output_nr) gets hooked up to the next input in the grad_fn
  list<int> gradient_fn_inputs;


  // OUTPUTS
  // one for each output of f'
  // which routes the gradients produced to the input_port of the n'th input
  // This is not in 1-1 correspondence with inputs because not all inputs require grad
  list<int> outputs;

  int num_temporary_outputs; // outputs we created from f that were not originally there

  // EXAMPLE, m does not require grad, n is determined to not require grad either (e.g. because it only depends on m)
  // y, n, t = f(x, m)

  // f':
  // in this case x, t and y were required temporaries and gradients for y ant t are required
  // dx = f'(x, t, y, dy, dt) // no dm  or dn because they do not require gradient

  // captured_inputs = {I0, O2, O1}
  //                    x   t   y
  // gradient_fn_inputs = {0, 2}
  //                       dy dt
  // i.e. connect grad_fn of y and t variables produced by f, with y's output_nr = 0 and t's output_nr = 1

  // outputs = {0}
  //            dx
  // i.e. connect next_function[0] of grad_fn to x's (grad_fn, output_nr).
  // num_temporary_outputs = 1; (t was produced for backward)
};

class Capture {
  enum {Input, Output} kind;
  int index;
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


// concern 1 mitigations:
// any graph executor that is running a gradient can assume variables will be the same size
// and not put that in the key since the sizes of grad_x is determined by x.
// However, grad/defined/strides may all be valid to change and we technically have to check.

// concern 2 mitigations
// in the unlikely event that these concats are costly, we can use a
// 'rope'-like structure (https://en.wikipedia.org/wiki/Rope_(data_structure))
// to avoid actually performing the concat.
// however, we almost always iterate this list at least once, so it is not clear
// that avoiding the concat is worthwhile anyway.

*/
