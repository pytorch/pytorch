#pragma once
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/function_schema.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/named_value.h"
#include "torch/csrc/jit/source_range.h"

#include <torch/csrc/api/include/torch/detail/ordered_dict.h>

#include <ATen/core/ArrayRef.h>
#include <ATen/core/optional.h>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// This file contains classes which assist in desugaring Python style
// modules and their methods into flattened graphs which don't have any
// function calls.

namespace torch { namespace jit { namespace script {

// A method in a module, e.g. f in:
//
// class M(ScriptModule):
//   @script_method
//   def f(self, x):
//     ...
// Note: because Method/Module are exposed to python these
// classes use python method naming conventions

struct Method {
  Method(std::string name, bool optimize,
         std::shared_ptr<Graph> graph,
         std::vector<at::Tensor*> initial_members,
         std::function<void(Method&)> method_creator)
  : name_(std::move(name))
  , graph_(std::move(graph))
  , optimize(optimize)
  , member_inputs(std::move(initial_members))
  , method_creator(method_creator) {
    JIT_ASSERT(graph_->inputs().size() >= member_inputs.size());
    int i = graph_->inputs().size() - member_inputs.size();
    for(at::Tensor* member : member_inputs) {
      member_input_index[member] = i++;
    }
  }

  void run(Stack & stack) {
    for(at::Tensor* tp : member_inputs) {
      stack.push_back(*tp);
    }
    get_executor().run(stack);
  }
  std::shared_ptr<Graph> graph_for(const Stack& inputs) {
    return get_executor().graphFor(inputs);
  }
  std::shared_ptr<Graph> graph() const {
    return graph_;
  }

  const std::string & name() const {
    return name_;
  }
  // emit a function call by inlining the callees Graph into this one
  // adding any extra parameters necessary to do this call

  // defined here to keep details of member_input handling confined to this class
  std::vector<Value*> emit_call_to(SourceRange loc, Method & callee, ArrayRef<NamedValue> args, ArrayRef<NamedValue> kwargs);
  // if this isn't yet defined, run its method_creator function
  void ensure_defined();


  size_t num_inputs() const {
    return graph()->inputs().size() - member_inputs.size();
  }
  Value * get_or_add_parameter(at::Tensor* slot) {
    auto it = member_input_index.find(slot);
    if(it != member_input_index.end()) {
      return graph()->inputs().at(it->second);
    }
    // add it as a new parameter
    member_inputs.push_back(slot);
    member_input_index[slot] = graph()->inputs().size();
    return graph()->addInput();
  }

  std::shared_ptr<Graph> propagate_shapes(std::vector<at::Tensor> inputs, bool with_grad=false) {
    auto retval = graph_->copy();
    Stack stack;
    stack.reserve(inputs.size() + member_inputs.size());
    for (at::Tensor & i : inputs) {
      stack.emplace_back(std::move(i));
    }
    for (at::Tensor* inp : member_inputs) {
      stack.push_back(*inp);
    }
    PropagateInputShapes(*retval, ArgumentSpec(with_grad, std::move(stack)));
    return retval;
  }

  std::shared_ptr<Graph> propagate_and_assign_input_and_output_shapes(std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs, bool with_grad=false, bool propagate=true) {
    auto retval = graph_->copy();
    for (auto inp : member_inputs) {
      inputs.push_back(*inp);
    }
    if (propagate) {
      PropagateInputShapes(*retval, ArgumentSpec(with_grad, fmap<IValue>(inputs)));
    }
    JIT_ASSERT(retval->inputs().size() == inputs.size());
    for (size_t i=0; i < retval->inputs().size(); ++i) {
      auto scalar_type = inputs[i].type().scalarType();
      auto sizes = inputs[i].sizes();
      auto type = torch::jit::TensorType::create(scalar_type, -1, sizes);
      retval->inputs()[i]->setType(type);
    }
    JIT_ASSERT(retval->outputs().size() == outputs.size());
    for (size_t i=0; i < retval->outputs().size(); ++i) {
      auto scalar_type = outputs[i].type().scalarType();
      auto sizes = outputs[i].sizes();
      auto type = torch::jit::TensorType::create(scalar_type, -1, sizes);
      retval->outputs()[i]->setType(type);
    }
    return retval;
  }

  std::vector<at::Tensor*> params() {
    return member_inputs;
  }

  Method& setSchema(FunctionSchema schema_) {
    schema.reset(new FunctionSchema(std::move(schema_)));
    return *this;
  }

  const FunctionSchema& getSchema() const {
    AT_ASSERT(schema != nullptr);
    return *schema;
  }

  std::string prettyPrintSchema() const {
    JIT_ASSERT(schema);
    std::stringstream ss;
    ss << *schema;
    return ss.str();
  }

  GraphExecutorState getDebugState() {
    return get_executor().getDebugState();
  }

private:
  std::string name_;
  std::shared_ptr<Graph> graph_; // for debugging and for inlining
  bool optimize;

  GraphExecutor& get_executor() {
    std::call_once(executor_init, [&]{
      executor = GraphExecutor(graph(), optimize);
    });
    return executor;
  }

  GraphExecutor executor; // for execution
  // member_inputs are a list of additional arguments appended to graph that are
  // inputs that come from the members of the Module or its submodules.
  // each is a pointer to a slot in the module that owns this parameter
  // parameters and submodules can only be _added_ to script Modules to ensure
  // these pointers always stay valid
  std::vector<at::Tensor*> member_inputs;

  // map from a at::Tensor* in member_inputs to the offset it appears at
  // in graph. used to accelerate get_or_add_parameter
  std::unordered_map<at::Tensor*, size_t> member_input_index;

  // TODO: support that case where we allow _writes_ to parameters from
  // compiled functions.
  // This requires more sophisticated tracking of ssa values in Graphs so that
  // stores to all modules can be lifted to the end of a graph execution.
  // It also adds more complexity to adding actual module invocations
  // to the executor, so currently it is not done.
  // std::vector<at::Tensor*> member_outputs;

  std::once_flag executor_init;

  // an optional function that actually creates the method when emit_call_to(this,...)
  // is first called.
  // this is used by the compiler so that it can construct methods out of order
  std::function<void(Method&)> method_creator;

  // if absent, then we generate a default schema based on the graph
  std::unique_ptr<FunctionSchema> schema;
};

struct Module;

struct NamedModule {
  std::string name;
  std::shared_ptr<Module> module;
};

struct NamedParameter {
  NamedParameter(std::string name, at::Tensor tensor, bool is_buffer)
  : name(std::move(name))
  , is_buffer(is_buffer)
  , parameter(new at::Tensor(std::move(tensor))) {}

  const std::string name;
  bool is_buffer; // buffers are part of the module state but
                        // are not modified by optimizers during SGD
  at::Tensor* slot() const {
    return parameter.get();
  }
private:
  // the extra level of indirection allows Methods to safely store pointers
  // to the slots where parameters are kept while also allow parameters
  // to be reassigned
  std::unique_ptr<at::Tensor> parameter;
};

struct Module {
  TH_DISALLOW_COPY_AND_ASSIGN(Module);
  Module()
  : modules("Module")
  , parameters("Parameter")
  , methods("Method")
  , optimize(true) {}

  // note this doesn't change the flags of existing methods just ones
  // added afterward.
  void set_optimized(bool o) {
    optimize = o;
  }

  void register_parameter(const std::string & name, autograd::Variable v, bool is_buffer) {
    if(auto p = parameters.find(name)){
      *p->slot() = v;
      p->is_buffer = is_buffer;
      return;
    }
    parameters.insert(name, NamedParameter(name, std::move(v), is_buffer));
  }
  void register_module(const std::string& name, std::shared_ptr<Module> module) {
    modules.insert(name, {name, std::move(module)});
  }

  Method& create_method(const std::string & name, std::shared_ptr<Graph> graph, std::vector<at::Tensor*> member_inputs) {
    JIT_ASSERT(graph);
    std::unique_ptr<Method> method(new Method(name, optimize, std::move(graph), std::move(member_inputs), nullptr));
    return *methods.insert(name, std::move(method));
  }

  Method& create_method(const std::string & name, std::function<void(Method&)> creator) {
    std::unique_ptr<Method> method(new Method(name, optimize, std::make_shared<Graph>(), {}, creator));
    return *methods.insert(name, std::move(method));
  }

  at::Tensor* parameter_slot(const std::string & name) const {
    return parameters.get(name).slot();
  }

  void set_parameter(const std::string & name, at::Tensor v) {
    *parameter_slot(name) = std::move(v);
  }

  autograd::Variable get_parameter(const std::string& name) const {
    return autograd::as_variable_ref(*parameter_slot(name));
  }

  // each module owns its method. The reference returned here
  // is guarenteed to stay valid until this module has been destroyed
  Method& get_method(const std::string& name) const {
    return *methods.get(name);
  }

  std::shared_ptr<Module> get_module(const std::string& name) const {
    return modules.get(name).module;
  }

  const torch::detail::OrderedDict<std::string, NamedModule>& get_modules() const {
    return modules;
  }
  const torch::detail::OrderedDict<std::string, NamedParameter>& get_parameters() const {
    return parameters;
  }
  const torch::detail::OrderedDict<std::string, std::unique_ptr<Method>>& get_methods() const {
    return methods;
  }

  NamedParameter* find_parameter(const std::string& name) {
    return parameters.find(name);
  }
  NamedModule* find_module(const std::string& name) {
    return modules.find(name);
  }
  Method* find_method(const std::string& name) {
    if (auto* pm = methods.find(name)) {
      return pm->get();
    }
    return nullptr;
  }

  void save(const std::string& filename);

 private:

  // invariant: to ensure member_inputs of Methods stay valid,
  // it is only legal to _add_ new modules and parameters.
  // removing them will allow member_inputs to point to invalid parameters
  // no such restriction exists for methods
  torch::detail::OrderedDict<std::string, NamedModule> modules;
  torch::detail::OrderedDict<std::string, NamedParameter> parameters;
  torch::detail::OrderedDict<std::string, std::unique_ptr<Method>> methods;
  bool optimize;
};

}}}
