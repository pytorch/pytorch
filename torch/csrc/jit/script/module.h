#pragma once
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/autograd/variable.h"
#include <ATen/optional.h>

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
         std::vector<at::Tensor*> initial_members)
  : name_(std::move(name))
  , graph_(std::move(graph))
  , optimize(optimize)
  , member_inputs(std::move(initial_members)) {
    JIT_ASSERT(graph_->inputs().size() >= member_inputs.size());
    int i = graph_->inputs().size() - member_inputs.size();
    for(at::Tensor* member : member_inputs) {
      member_input_index[member] = i++;
    }
  }

  variable_tensor_list run(variable_tensor_list && inputs) {
    std::call_once(executor_init, [&]{
      executor = GraphExecutor(graph_, optimize);
    });
    for(auto tp : member_inputs) {
      inputs.push_back(*tp);
    }
    return executor.run(std::move(inputs));
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
  std::vector<Value*> emit_call_to(Method & callee, ArrayRef<Value*> inputs);

  size_t num_inputs() const {
    return graph_->inputs().size() - member_inputs.size();
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
private:
  std::string name_;
  std::shared_ptr<Graph> graph_; // for debugging and for inlining
  bool optimize;
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

// simple ordered dict used only in Module
// contains only the minimum necessary functionality for Module
template<typename T>
struct OrderedDict {
  OrderedDict() {}
  T& insert(const std::string& name,  T&& value, const char* what) {
    if(index_.count(name) != 0) {
      std::stringstream ss;
      ss << "module " << what << "'" << name << "' already defined.";
      throw std::runtime_error(ss.str());
    }
    values_.push_back(std::move(value));
    index_[name] = values_.size() - 1;
    return values_.back();
  }
  at::optional<T&> find(const std::string& str) {
    auto it = index_.find(str);
    if(it == index_.end())
      return at::nullopt;
    return at::optional<T&>(values_.at(it->second));
  }
  at::optional<const T&> find(const std::string& str) const {
    auto it = index_.find(str);
    if(it == index_.end())
      return at::nullopt;
    return at::optional<const T&>(values_.at(it->second));
  }
  T& get(const std::string& name, const char * what) {
    if(auto v = find(name)) {
      return *v;
    }
    std::stringstream ss;
    ss << "module " << what << "'" << name << "' is not defined.";
    throw std::runtime_error(ss.str());
  }
  const T& get(const std::string& name, const char * what) const {
    if(auto v = find(name)) {
      return *v;
    }
    std::stringstream ss;
    ss << "module " << what << "'" << name << "' is not defined.";
    throw std::runtime_error(ss.str());
  }
  const std::vector<T>& values() const {
    return values_;
  }
private:
  std::unordered_map<std::string, size_t> index_;
  std::vector<T> values_;
};

struct Module : public std::enable_shared_from_this<Module> {
  TH_DISALLOW_COPY_AND_ASSIGN(Module);
  Module(bool optimize)
  : optimize(optimize) {}

  void register_parameter(const std::string & name, autograd::Variable v, bool is_buffer) {
    if(auto p = parameters.find(name)){
      *p->slot() = v;
      p->is_buffer = is_buffer;
      return;
    }
    parameters.insert(name, NamedParameter(name, std::move(v), is_buffer), "parameter");
  }
  void register_module(const std::string& name, std::shared_ptr<Module> module) {
    modules.insert(name, {name, std::move(module)}, "module");
  }

  Method& create_method(const std::string & name, std::shared_ptr<Graph> graph = nullptr, std::vector<at::Tensor*> member_inputs = {}) {
    if(!graph)
      graph = std::make_shared<Graph>();
    std::unique_ptr<Method> method(new Method(name, optimize, std::move(graph), std::move(member_inputs)));

    return *methods.insert(name, std::move(method), "method");
  }

  at::Tensor* parameter_slot(const std::string & name) const {
    return parameters.get(name, "parameter").slot();
  }

  void set_parameter(const std::string & name, at::Tensor v) {
    *parameter_slot(name) = std::move(v);
  }

  autograd::Variable get_parameter(const std::string& name) const {
    return static_cast<autograd::Variable&>(*parameter_slot(name));
  }

  // each module owns its method. The reference returned here
  // is guarenteed to stay valid until this module has been destoryed
  Method& get_method(const std::string& name) const {
    return *methods.get(name, "method");
  }

  std::shared_ptr<Module> get_module(const std::string& name) const {
    return modules.get(name, "module").module;
  }

  const std::vector<NamedModule>& get_modules() const {
    return modules.values();
  }
  const  std::vector<NamedParameter>& get_parameters() const {
    return parameters.values();
  }


  at::optional<NamedParameter&> find_parameter(const std::string& name) {
    return parameters.find(name);
  }
  at::optional<NamedModule&> find_module(const std::string& name) {
    return modules.find(name);
  }
  at::optional<Method&> find_method(const std::string& name) {
    if(auto pm = methods.find(name))
      return at::optional<Method&>(**pm);
    return at::nullopt;
  }

private:

  // invariant: to ensure member_inputs of Methods stay valid,
  // it is only legal to _add_ new modules and parameters.
  // removing them will allow member_inputs to point to invalid parameters
  // no such restriction exists for methods
  OrderedDict<NamedModule> modules;
  OrderedDict<NamedParameter> parameters;
  OrderedDict<std::unique_ptr<Method>> methods;
  bool optimize;
};

}}}
