#pragma once
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/autograd/variable.h"

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
struct Method {
  Method(std::string name, bool optimize)
  : name_(std::move(name))
  , graph_(std::make_shared<Graph>())
  , optimize(optimize) {}
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
  // each is a pointer to a slot in the module that owns this Method or a submethod
  // of the module.
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
  NamedParameter(std::string name, at::Tensor tensor)
  : name(std::move(name)), parameter(new at::Tensor(std::move(tensor))) {}

  std::string name;
  at::Tensor* slot() const {
    return parameter.get();
  }
private:
  // the extra level of indirection allows Methods to safely store pointers
  // to the slots where parameters are kept while also allow parameters
  // to be reassigned
  std::unique_ptr<at::Tensor> parameter;
};

struct NamedMember {
  enum Kind { Module, Parameter, Method, None };
  // note: None is used to report undefined attributes;
  Kind kind;
  size_t offset;

  static const char * kind_string(Kind kind) {
    switch(kind) {
      case Module: return "module";
      case Parameter: return "parameter";
      case Method: return "method";
      case None: return "none";
      default: return "unknown";
    }
  }
};

struct Module : public std::enable_shared_from_this<Module> {
  TH_DISALLOW_COPY_AND_ASSIGN(Module);
  Module(bool optimize)
  : optimize(optimize) {}

  void register_parameter(const std::string & name, at::Tensor v) {
    parameters.push_back(NamedParameter(name, std::move(v)));
    add_member(name, NamedMember::Parameter, parameters.size() - 1);
  }
  void register_or_set_parameter(const std::string & name, autograd::Variable v) {
    if(find_attribute(name) == NamedMember::Parameter) {
      set_parameter(name, v);
    } else {
      register_parameter(name, v);
    }
  }
  void register_module(const std::string& name, std::shared_ptr<Module> module) {
    JIT_ASSERT(module);
    modules.push_back(NamedModule {name, std::move(module)});
    add_member(name, NamedMember::Module, modules.size() - 1);
  }

  Method& create_method(const std::string & name) {
    methods.emplace_back(new Method(name, optimize));
    add_member(name, NamedMember::Method, methods.size() - 1);
    return *methods.back();
  }

  at::Tensor* parameter_slot(const std::string & name) {
    return parameters.at(find_member(name, NamedMember::Parameter)).slot();
  }

  void set_parameter(const std::string & name, at::Tensor v) {
    *parameter_slot(name) = std::move(v);
  }

  at::Tensor get_parameter(const std::string& name) {
    return *parameter_slot(name);
  }

  // each module owns its method. The reference returned here
  // is guarenteed to stay valid until this module has been destoryed
  Method& get_method(const std::string& name) const {
    return *methods.at(find_member(name, NamedMember::Method));
  }

  std::shared_ptr<Module> get_module(const std::string& name) const {
    auto loc = find_member(name, NamedMember::Module);
    return modules.at(loc).module;
  }

  NamedMember::Kind find_attribute(const std::string& name) {
    auto it = members.find(name);
    if(it == members.end())
      return NamedMember::None;
    return it->second.kind;
  }

  void dump() const {
    for(auto entry : members) {
      std::cout << entry.first << ": " << NamedMember::kind_string(entry.second.kind) << "\n";
    }
  }

private:
  size_t find_member(const std::string& name, NamedMember::Kind kind) const  {
    auto it = members.find(name);
    if(it == members.end()) {
      std::stringstream ss;
      ss << "unknown " << NamedMember::kind_string(kind) << " '" << name << "'";
      throw std::runtime_error(ss.str());
    }
    if(it->second.kind != kind) {
      std::stringstream ss;
      ss << "Expected attribute '" << name << "' to be a "
        << NamedMember::kind_string(kind) << " but found "
        << NamedMember::kind_string(it->second.kind);
      throw std::runtime_error(ss.str());
    }
    JIT_ASSERT(it != members.end() && it->second.kind == kind);
    return it->second.offset;
  }
  void add_member(const std::string& name, NamedMember::Kind kind, size_t offset) {
    auto it = members.find(name);
    if(it != members.end()) {
      std::stringstream ss;
      ss << "attempting to add " << NamedMember::kind_string(kind) << " '" << name << "' but Module already contains "
      << NamedMember::kind_string(it->second.kind) << " '" << name << "'";
      throw std::runtime_error(ss.str());
    }
    members[std::move(name)] = NamedMember { kind, offset };
  }
  // invariant: to ensure member_inputs of Methods stay valid,
  // it is only legal to _add_ new modules and parameters.
  // removing them will allow member_inputs to point to invalid parameters
  // no such restriction exists for methods
  std::vector<NamedModule> modules;
  std::vector<NamedParameter> parameters;
  std::vector<std::unique_ptr<Method>> methods;

  std::unordered_map<std::string, NamedMember> members;
  bool optimize;
};

}}}
