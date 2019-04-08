#pragma once
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/named_value.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/script/slot.h>
#include <torch/csrc/jit/source_range.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/api/include/torch/ordered_dict.h>
#include <torch/csrc/utils/memory.h>
#include <torch/csrc/jit/script/compilation_unit.h>

#include <ATen/core/function_schema.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

// This file contains classes which assist in desugaring Python style
// modules and their methods into flattened graphs which don't have any
// function calls.

namespace torch {
namespace jit {
namespace script {

using ::c10::Argument;
using ::c10::FunctionSchema;
// Map which stores filename to content.
using ExtraFilesMap = std::unordered_map<std::string, std::string>;


using ModulePtr = c10::intrusive_ptr<c10::ivalue::Object>;
// A method in a module, e.g. f in:
//
// class M(ScriptModule):
//   @script_method
//   def f(self, x):
//     ...
// Note: because Method/Module are exposed to python these
// classes use python method naming conventions

struct Module;

using ModuleLookup =
    std::function<std::shared_ptr<Module>(const std::vector<std::string>&)>;

struct Method {
  Method(
      Module* owner,
      Function* function,
      std::vector<Slot> initial_members)
      : owner_(owner),
        function_(function),
        initial_ivalues_(std::move(initial_members)) {
    AT_ASSERT(function->num_inputs() >= initial_ivalues_.size());
  }

  // the module that contains this method.
  Module& owner() const {
    return *owner_;
  }

  void run(Stack& stack) {
    for (auto input : initial_ivalues_) {
      push(stack, input.value());
    }
    function_->run(stack);
  }
  void run(Stack&& stack) {
    run(stack);
  }

  IValue operator()(std::vector<IValue> stack) {
    for (auto input : initial_ivalues_) {
      push(stack, input.value());
    }
    return (*function_)(std::move(stack));
  }

  const std::vector<Slot>& initial_ivalues() const {
    return initial_ivalues_;
  }


  // proxies for underlying unbound Function
  std::shared_ptr<Graph> graph_for(Stack inputs) {
    for (auto tp : initial_ivalues_) {
      inputs.emplace_back(tp.value());
    }
    return function_->get_executor().graphFor(inputs);
  }

  TORCH_API std::shared_ptr<Graph> graph() const {
    return function_->graph();
  }

  TORCH_API const std::string& name() const {
    return function_->name();
  }

  size_t num_inputs() const {
    return function_->num_inputs() - initial_ivalues_.size();
  }

  FunctionSchema getSchema() const {
    // we are required to slice out the slot inputs from the schema
    // we can't cache this because setSchema on the underlying function
    // will change the underlying schema
    auto sliced = ArrayRef<Argument>(function_->getSchema().arguments()).slice(0, num_inputs());
    return function_->getSchema().withArguments(sliced.vec());
  }

  GraphExecutor& get_executor() {
    return function_->get_executor();
  }

  Function& function() const {
    return *function_;
  }
 private:
  // Methods are uniqued onwed by a single module. This raw pointer allows
  // looking up the module.
  Module* owner_;

  //Underlying unbound function
  Function* function_;

  // parameters and attributes loaded from the Module and appending
  // before calling function_
  std::vector<Slot> initial_ivalues_;

};

struct Module;

struct Module {
  TH_DISALLOW_COPY_AND_ASSIGN(Module);
  Module()
      : name_("__main__"),
        module_value_(c10::ivalue::Object::create(
            ClassType::createModuleType(std::make_shared<CompilationUnit>()),
            0)) {}

  const std::string& name() const {
    return name_;
  }

  // note this doesn't change the flags of existing methods just ones
  // added afterward.
  void set_optimized(bool o) {
    class_cu().set_optimized(o);
  }

  bool is_optimized() const {
    return class_cu().is_optimized();
  }

  IValue forward(std::vector<IValue> inputs) {
    return get_method("forward")(std::move(inputs));
  }

  void register_buffer(const std::string& name, autograd::Variable v) {
    if (auto b = find_attribute(name)) {
      AT_ASSERT(b->type()->isSubtypeOf(TensorType::get()));
      b->setValue(v);
      return;
    }
    insert(
        name,
        attributes_,
        EntityType::ATTRIBUTE,
        appendSlot(name, TensorType::get(),std::move(v)));
  }

  void register_parameter(
      const std::string& name,
      autograd::Variable v,
      bool is_buffer) {
    if (is_buffer) {
      register_buffer(name, std::move(v));
      return;
    }
    if (auto p = find_parameter(name)) {
      p->setValue(v);
      return;
    }
    insert(
        name,
        parameters_,
        EntityType::PARAMETER,
        appendSlot(name, TensorType::get(), std::move(v)));
  }
  void register_attribute(
      const std::string& name,
      const TypePtr type,
      IValue ivalue) {
    insert(name, attributes_, EntityType::ATTRIBUTE, appendSlot(name, type, ivalue));
  }
  void register_module(
      const std::string& name,
      std::shared_ptr<Module> module) {
    // We would like to enable more stringent error checking at this point,
    // but because script functions are considered modules, it is possible
    // to hit this situation without knowing it. For now this is disabled
    // until a later PR that distinguishes script functions from script modules.
    // See TestScript.test_submodule_twice for example failure
    // if (module->parent_) {
    //   AT_WARN(
    //       "Attempting to assign submodule '",
    //       name,
    //       "' but it is already a submodule of another ScriptModule '", module->parent_->name(), "'",
    //       " Modules of this form do not import and export correctly. This use is deprecated and may be"
    //       " removed in a future version.");
    // }
    module->parent_ = this;
    module->name_ = name;
    appendSlot(name, module->module_value_->type(), module->module_value_);
    insert(name, modules_, EntityType::MODULE, std::move(module));
  }


  Slot parameter_slot(const std::string& name) const {
    return parameters_[get_offset(name, EntityType::PARAMETER)];
  }

  void set_parameter(const std::string& name, at::Tensor v) {
    parameter_slot(name).setValue(std::move(v));
  }

  autograd::Variable get_parameter(const std::string& name) const {
    return autograd::as_variable_ref(parameter_slot(name).value().toTensor());
  }

  IValue get_attribute(const std::string& name) const {
    return attributes_[get_offset(name, EntityType::ATTRIBUTE)].value();
  }

  autograd::Variable get_buffer(const std::string& name) const {
    return autograd::as_variable_ref(get_attribute(name).toTensor());
  }

  // each module owns its method. The reference returned here
  // is guarenteed to stay valid until this module has been destroyed
  Method& get_method(const std::string& name) const {
    if (Method* method = find_method(name)) {
      return *method;
    }
    // temporary: force the error message
    // once the on-demand creation of Method is removed, this code
    // can be removed as well
    get_offset(name, EntityType::METHOD);
    AT_ERROR("unreachable");
  }

  std::shared_ptr<Module> get_module(const std::string& name) const {
    return modules_[get_offset(name, EntityType::MODULE)];
  }

  c10::ArrayRef<std::shared_ptr<Module>> get_modules() const {
    return modules_;
  }
  c10::ArrayRef<Slot> get_parameters() const {
    return parameters_;
  }
  c10::ArrayRef<Slot> get_attributes() const {
    return attributes_;
  }
  const std::vector<std::unique_ptr<Method>>& get_methods() const {
    // force methods_ to be up to date by querying all
    // methods. This will go away when lowered_methods_ is deleted
    for(const auto& m : class_cu().get_functions()) {
      get_method(m->name());
    }
    return methods_;
  }

  Slot* find_parameter(const std::string& name) {
    auto offset = find_offset(name, EntityType::PARAMETER);
    return offset ? &parameters_[*offset] : nullptr;
  }
  Slot* find_attribute(const std::string& name) {
    auto offset = find_offset(name, EntityType::ATTRIBUTE);
    return offset ? &attributes_[*offset] : nullptr;
  }
  Slot* find_buffer(const std::string& name) {
    auto iv = find_attribute(name);
    if (iv && iv->type()->isSubtypeOf(TensorType::get())) {
      return iv;
    }
    return nullptr;
  }
  std::shared_ptr<Module> find_module(const std::string& name) {
    auto offset = find_offset(name, EntityType::MODULE);
    return offset ? modules_[*offset] : nullptr;
  }
  Method* find_method(const std::string& name) const {
    auto offset = find_offset(name, EntityType::METHOD);
    if (offset) {
      return methods_[*offset].get();
    }

    if (Function* fn = class_cu().find_function(name)) {
      return &const_cast<Module*>(this)->lower_first_class_method(fn);
    }

    return nullptr;
  }
  void apply(std::function<void(Module&)> fn) {
    for (auto& submod : get_modules()) {
      submod->apply(fn);
    }
    fn(*this);
  }
  /// Enables "training" mode.
  void train(bool on = true) {
    for (auto& submod : get_modules()) {
      submod->train(on);
    }
    register_buffer("training", torch::tensor(on ? 1 : 0, at::kLong));
  }
  /// Calls train(false) to enable "eval" mode.
  /// Do not override this method, override `train()` instead.
  void eval() {
    train(/*on=*/false);
  }
  /// True if the module is in training mode.
  bool is_training() {
    if (auto p = find_buffer("training")) {
      return p->value().toTensor().item<int64_t>() == 1;
    }
    // We are in training mode by default
    return true;
  }

  /// Recursively casts all parameters to the given `dtype` and `device`.
  ///
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  TORCH_API void to(
      at::Device device,
      at::ScalarType dtype,
      bool non_blocking = false);

  /// Recursively casts all parameters to the given dtype.
  ///
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  TORCH_API void to(at::ScalarType dtype, bool non_blocking = false);

  /// Recursively moves all parameters to the given device.
  ///
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  TORCH_API void to(at::Device device, bool non_blocking = false);

  /// Run a method from this module.
  ///
  /// For example:
  /// @code
  ///   IValue output = module->run("relu_script", a, b);
  /// @endcode
  ///
  /// To get a compile a module from a source string, see torch::jit::compile
  ///
  /// @param method_name The name of the method to run
  /// @param args Arguments to be passed to the method
  /// @return An IValue containing the return value (or values if it is a tuple)
  /// from the method
  template <typename... Types>
  IValue run_method(const std::string& method_name, Types&&... args) {
    return get_method(method_name)({IValue(std::forward<Types>(args))...});
  }

  void save(
      std::ostream& out,
      const ExtraFilesMap& extra_files = ExtraFilesMap());

  void save(
      const std::string& filename,
      const ExtraFilesMap& extra_files = ExtraFilesMap());

  void copy_into(
      ModuleLookup module_lookup,
      // parameter_remap is needed when a parent module uses a parameter of a
      // submodule
      std::unordered_map<Slot, Slot>& parameter_remap,
      std::vector<std::string> names = {}) const {
    auto curr = module_lookup(names);
    for (auto& param : get_parameters()) {
      curr->register_parameter(
          param.name(),
          param.value().toTensor(),
          /*is_buffer=*/false);
      parameter_remap[param] = curr->parameter_slot(param.name());
    }
    for (auto& attr : get_attributes()) {
      if (!attr.type()->isSubtypeOf(TensorType::get())) {
        continue;
      }
      curr->register_buffer(attr.name(), attr.value().toTensor());
      parameter_remap[attr] = *curr->find_buffer(attr.name());
    }
    for (auto& mod : get_modules()) {
      names.push_back(mod->name());
      // Submodules must be translated first, otherwise parameter_remap entries
      // will not be filled in for methods of this module.
      mod->copy_into(module_lookup, parameter_remap, names);
      names.pop_back();
    }

    for (auto& fn : class_cu().get_functions()) {
      curr->class_cu().clone_function(*fn);
    }
  }

  enum class EntityType { MODULE, PARAMETER, ATTRIBUTE, METHOD };

  at::optional<EntityType> kind_of(const std::string& name) const {
    // force lazy creation of Method if needed
    // remove once lowered_methods_ is removed.
    find_method(name);

    auto it = dict_.find(name);
    if (it == dict_.end())
      return at::nullopt;
    return it->second.type;
  }

  ModulePtr module_object() const {
    return module_value_;
  }
  CompilationUnit& class_compilation_unit() {
    return module_object()->type()->compilation_unit();
  }
  CompilationUnit& lowered_methods() const {
    return lowered_methods_;
  }


  void _define_lowered(const std::vector<Def>& definitions,
                       const std::vector<Resolver>& resolvers);
  void _define_lowered(const std::string& src,
                       const Resolver& resolver);

  Method& _define_lowered(
      std::string name,
      std::shared_ptr<Graph> graph,
      std::vector<Slot> slots);

 private:
  Method& _create_lowered_method(
     Function* func,
     std::vector<Slot> member_inputs);

  Method& lower_first_class_method(Function* fn);
  void lift_lowered_method(Method& fn);
  void lift_lowered_methods(size_t start);

  void to_impl(
      const c10::optional<at::Device>& device,
      const c10::optional<at::ScalarType>& dtype,
      bool non_blocking);

  static const char* toString(EntityType t) {
    switch (t) {
      case EntityType::MODULE:
        return "module";
      case EntityType::PARAMETER:
        return "parameter";
      case EntityType::ATTRIBUTE:
        return "attribute";
      case EntityType::METHOD:
        return "method";
    }
    return nullptr;
  }

  struct Entry {
    EntityType type;
    size_t offset;
  };

  size_t get_offset(const std::string& name, EntityType expected_type) const {
    auto it = dict_.find(name);
    if (it == dict_.end()) {
      AT_ERROR(toString(expected_type), " '", name, "' is not defined.");
    }
    if (it->second.type != expected_type) {
      AT_ERROR(
          "The field '",
          name,
          "' is a ",
          toString(it->second.type),
          " but this call is"
          " trying to use it as a ",
          toString(expected_type));
    }
    return it->second.offset;
  }
  at::optional<size_t> find_offset(
      const std::string& name,
      EntityType expected_type) const {
    auto it = dict_.find(name);
    if (it == dict_.end() || it->second.type != expected_type) {
      return at::nullopt;
    }
    return it->second.offset;
  }

  template <typename T>
  T& insert(
      const std::string& name,
      std::vector<T>& list,
      EntityType type,
      T value) {
    auto it = dict_.find(name);
    if (it != dict_.end()) {
      if (type != it->second.type) {
        AT_ERROR(
            "attempting to add ",
            toString(type),
            " '",
            name,
            "' but it already exists as a ",
            toString(it->second.type));
      } else {
        AT_ERROR(toString(type), " '", name, "' already defined.");
      }
    }
    dict_[name] = Entry{type, list.size()};
    list.emplace_back(std::move(value));
    return list.back();
  }

  // add a new entry to the singleton object that represents this
  // Module as a first-class value in code, and update the corresponding
  // ClassType to match.
  Slot appendSlot(const std::string& name, TypePtr typ, IValue value) {
    const ClassTypePtr& type = module_value_->type();
    type->addAttribute(name, std::move(typ));
    auto slot_index = type->getAttributeSlot(name);
    module_value_->setSlot(slot_index, std::move(value));
    return Slot(module_value_, slot_index);
  }

  CompilationUnit& class_cu() {
    return module_value_->type()->compilation_unit();
  }
  const CompilationUnit& class_cu() const {
    return module_value_->type()->compilation_unit();
  }

  // modules have a single namespace, but spread over 4 different concepts:
  // parameters, attributes, methods, and sub-modules
  // we store individual lists of each concept, and a single map to
  // unify the namespace and ensure fast lookup

  // invariant: to ensure initial_ivalues of Methods stay valid,
  // it is only legal to _add_ new modules and parameters.
  // removing them will allow initial_ivalues to point to invalid parameters
  // no such restriction exists for methods
  std::vector<std::shared_ptr<Module>> modules_;
  std::vector<Slot> parameters_;
  std::vector<Slot> attributes_;
  std::vector<std::unique_ptr<Method>> methods_;

  std::unordered_map<std::string, Entry> dict_;
  std::string name_;


  ModulePtr module_value_;


  // back reference to parent of this Module if present
  Module* parent_ = nullptr;


  // Currently we are in a transitionary state
  // where we construct such first class functions but we lower them
  // to a form where the modules does not exist before execution.

  // So each Method is actually stored twice once in first-class Module
  // form and once in lowered form.

  // first-class: module_value_->type().compilation_unit() holds Functions that treat
  // modules as first class.

  // lowered: In this lowered form, all the attributes/parameters are appended
  // as additional inputs. lowered_methods_ holds this lowered form
  // mutable because it is a cache for class_cu() methods
  mutable CompilationUnit lowered_methods_;
};

static void setInputTensorTypes(Graph& g, const Stack& stack) {
  AT_ASSERT(stack.size() == g.inputs().size());
  for (size_t i = 0; i < stack.size(); ++i) {
    g.inputs().at(i)->setType(
        DimensionedTensorType::create(stack.at(i).toTensor()));
  }
}

inline std::shared_ptr<Graph> propagate_shapes(Graph& graph,
    const std::vector<at::Tensor>& inputs,
    const std::vector<Slot>& initial_ivalues,
    bool with_grad = false) {
  auto retval = graph.copy();
  Stack stack;
  stack.reserve(inputs.size() + initial_ivalues.size());
  for (const at::Tensor& i : inputs) {
    stack.emplace_back(std::move(i));
  }
  for (const Slot& inp : initial_ivalues) {
    stack.push_back(inp.value());
  }
  setInputTensorTypes(*retval, stack);
  PropagateInputShapes(retval);
  return retval;
}

inline std::shared_ptr<Graph> propagate_and_assign_input_and_output_shapes(
    Graph& graph,
    std::vector<at::Tensor> inputs,
    const std::vector<Slot>& initial_ivalues,
    std::vector<at::Tensor> outputs,
    bool with_grad = false,
    bool propagate = true) {
  auto retval = graph.copy();
  for (auto inp : initial_ivalues) {
    if (inp.value().isTensor()) {
      inputs.push_back(inp.value().toTensor());
    }
  }
  if (propagate) {
    setInputTensorTypes(*retval, fmap<IValue>(inputs));
    PropagateInputShapes(retval);
  }
  AT_ASSERT(retval->inputs().size() == inputs.size());
  for (size_t i = 0; i < retval->inputs().size(); ++i) {
    auto scalar_type = inputs[i].scalar_type();
    auto sizes = inputs[i].sizes();
    auto type =
        torch::jit::CompleteTensorType::create(scalar_type, at::kCPU, sizes);
    retval->inputs()[i]->setType(type);
  }
  at::ArrayRef<Value*> output_values = retval->outputs();
  // patch this to still work if we are returning a tuple of multiple values
  if (output_values.at(0)->type()->kind() == TupleType::Kind) {
    AT_ASSERT(output_values.at(0)->node()->kind() == prim::TupleConstruct);
    output_values = output_values.at(0)->node()->inputs();
  }
  AT_ASSERT(output_values.size() == outputs.size());
  for (size_t i = 0; i < retval->outputs().size(); ++i) {
    auto scalar_type = outputs[i].scalar_type();
    auto sizes = outputs[i].sizes();
    auto type =
        torch::jit::CompleteTensorType::create(scalar_type, at::kCPU, sizes);
    output_values[i]->setType(type);
  }
  return retval;
}

} // namespace script
} // namespace jit
} // namespace torch
