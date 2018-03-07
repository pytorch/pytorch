#pragma once
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace jit { namespace script {

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
  std::vector<Value*> emit_call_to(Method & callee, ArrayRef<Value*> inputs) {
    JIT_ASSERT(!executor);
    auto fn = callee.graph();
    JIT_ASSERT(inputs.size() == callee.num_inputs());
    std::unordered_map<Value*, Value*> value_map;
    auto value_map_func = [&](Value* v) { return value_map.at(v); };
    // actual inputs
    for (size_t i = 0; i < inputs.size(); ++i) {
      value_map[fn->inputs()[i]] = inputs[i];
    }
    // parameters to callee method (which become parameters to _this_ method
    // if they were not already)
    auto members_it = callee.member_inputs.begin();
    for(size_t i = inputs.size(); i < fn->inputs().size(); i++) {
      value_map[fn->inputs()[i]] = get_or_add_parameter(*members_it++);
    }
    for (auto* node : fn->nodes()) {
      auto* new_node =
          graph_->insertNode(graph_->createClone(node, value_map_func));
      for (size_t i = 0; i < node->outputs().size(); ++i) {
        value_map[node->outputs()[i]] = new_node->outputs()[i];
      }
    }

    std::vector<Value*> outputs;
    for (auto* output : fn->outputs()) {
      outputs.push_back(value_map_func(output));
    }
    return outputs;
  }
  size_t num_inputs() const {
    return graph_->inputs().size() - member_inputs.size();
  }
  Value * get_or_add_parameter(at::Tensor* slot) {
    size_t first_parameter = graph()->inputs().size() - member_inputs.size();
    for(size_t i = 0; i < member_inputs.size(); i++) {
      // it is already a parameter
      if(member_inputs[i] == slot) {
        return graph()->inputs()[first_parameter + i];
      }
    }
    // add it as a new parameter
    member_inputs.push_back(slot);
    return graph()->addInput();
  }
private:
  std::string name_;
  std::shared_ptr<Graph> graph_; // for debugging and for inlining
  bool optimize;
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
  // to be reassign
  std::unique_ptr<at::Tensor> parameter;
};

struct NamedMember {
  enum Kind { Module, Parameter, Method, None};
  // note: None is used to report undefined attributes;
  Kind kind;
  size_t offset;

  static const char * kind_string(Kind kind) {
    switch(kind) {
      case Module: return "Module";
      case Parameter: return "Parameter";
      case Method: return "Method";
      case None: return "None";
      default: return "Unknown";
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
