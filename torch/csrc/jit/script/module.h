#pragma once
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/graph_executor.h"

namespace torch { namespace jit { namespace script {

struct Method {
  Method(std::string name, std::shared_ptr<Graph> graph, std::vector<at::Tensor*> member_inputs, bool optimize)
  : name_(std::move(name)), graph_(std::move(graph)), member_inputs(std::move(member_inputs)) {
    // TODO: we may want to lazily construct this since not all functions will
    // be called and creating executors takes time in optimization
    executor = GraphExecutor(graph_, optimize);
  }
  variable_tensor_list run(variable_tensor_list && inputs) {
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
private:
  std::string name_;
  std::shared_ptr<Graph> graph_; // for debugging and for inlining
  GraphExecutor executor; // for execution
  // member_inputs are a list of additional arguments append to graph that are
  // inputs that come from the members of the Module or its submodules.
  // each is a pointer to a slot in the module that owns this Method or a submethod
  // of the module.
  // parameters and submodules can only be _added_ to script Modules to ensure
  // these pointers always stay valid
  std::vector<at::Tensor*> member_inputs;

  // TODO: support that case where we allow _writes_ to parameters from
  // compiled functions.
  // This requires more sophisticated tracking of ssa values in Graphs so that
  // stores to all modules can be lifted to the end of a graph execution.
  // It also adds more complexity to adding actual module invocations
  // to the executor, so currently it is not done.
  // std::vector<at::Tensor*> member_outputs;
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
  // to be reassign
  std::unique_ptr<at::Tensor> parameter;
};

struct NamedMember {
  enum Kind { Module, Parameter, Method} kind;
  size_t offset;

  static const char * kind_string(Kind kind) {
    switch(kind) {
      case Module: return "Module";
      case Parameter: return "Parameter";
      case Method: return "Method";
      default: return "Unknown";
    }
  }
};

struct Module : public std::enable_shared_from_this<Module> {
  TH_DISALLOW_COPY_AND_ASSIGN(Module);
  Module(bool optimize = true) {}

  void register_parameter(const std::string & name, at::Tensor v) {
    parameters.push_back(NamedParameter(name, std::move(v)));
    add_member(name, NamedMember::Parameter, parameters.size() - 1);
  }
  void register_module(const std::string& name, std::shared_ptr<Module> module) {
    modules.push_back(NamedModule {name, std::move(module)});
    add_member(name, NamedMember::Module, modules.size() - 1);
  }

  void register_method(const std::string & name, std::shared_ptr<Graph> graph, std::vector<at::Tensor*> member_inputs) {
    methods.push_back(Method(name, std::move(graph), std::move(member_inputs), optimize));
    add_member(name, NamedMember::Method, methods.size() - 1);
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

  const Method& get_method(const std::string& name) const {
    return methods.at(find_member(name, NamedMember::Method));
  }

  std::shared_ptr<Module> get_module(const std::string& name) const {
    return modules.at(find_member(name, NamedMember::Module)).module;
  }

private:
  size_t find_member(const std::string& name, NamedMember::Kind kind) const  {
    auto it = members.find(name);
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
  std::vector<Method> methods;

  std::unordered_map<std::string, NamedMember> members;
  bool optimize;
};

}}}
