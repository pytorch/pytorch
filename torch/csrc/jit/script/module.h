#pragma once
#include <c10/util/Exception.h>
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
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/utils/memory.h>

#include <ATen/core/function_schema.h>
#include <ATen/core/qualified_name.h>
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
using ::c10::QualifiedName;
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

struct TORCH_API Method {
  Method(Module* owner, std::shared_ptr<Function> function);

  // the module that contains this method.
  Module& owner() const {
    return *owner_;
  }

  void run(Stack& stack);
  void run(Stack&& stack) {
    run(stack);
  }

  IValue operator()(std::vector<IValue> stack, const Kwargs& kwargs = Kwargs());

  std::shared_ptr<Graph> graph() const {
    return function_->graph();
  }

  const std::string& name() const {
    return function_->name();
  }

  size_t num_inputs() const {
    return function_->num_inputs();
  }

  GraphExecutor& get_executor() {
    return function_->get_executor();
  }

  Function& function() const {
    return *function_;
  }

  // Used for ONNX export. Return a tuple (graph, parameters) where
  // the last parameters.size() inputs to the graph are the trainable parameters
  // used in this method. The remaining inputs are the true inputs to the function.
  std::pair<std::shared_ptr<Graph>, std::vector<at::Tensor>> _lowered_graph();

 private:
  // Methods are uniqued onwed by a single module. This raw pointer allows
  // looking up the module.
  Module* owner_;

  // Underlying unbound function
  // This is the _lowered_ function and is different than the
  // first-class function in class_compilation_unit()
  std::shared_ptr<Function> function_;
};

struct Module;

struct TORCH_API Module {
  TH_DISALLOW_COPY_AND_ASSIGN(Module);
  Module(std::string class_name)
      : field_name_("__main__"),
        module_value_(c10::ivalue::Object::create(
            ClassType::create(
                QualifiedName(std::move(class_name)),
                std::make_shared<CompilationUnit>(),
                /*is_module=*/true),
            0)) {}
  Module() : Module("$Module") {}
  ~Module() {
    // ClassType own the compilation unit of their Functions, but each
    // Function has a self argument which owns the ClassType, created a
    // reference cycle. By dropping all the methods of the module's class
    // here we break the cycle.
    class_compilation_unit()->drop_all_functions();
  }
  const std::string& field_name() const {
    return field_name_;
  }

  // note this doesn't change the flags of existing methods just ones
  // added afterward.
  void set_optimized(bool o) {
    class_compilation_unit()->set_optimized(o);
  }

  bool is_optimized() const {
    return class_compilation_unit()->is_optimized();
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
        appendSlot(name, TensorType::get(), std::move(v)));
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
    insert(
        name,
        attributes_,
        EntityType::ATTRIBUTE,
        appendSlot(name, type, ivalue));
  }
  void register_module(
      const std::string& name,
      std::shared_ptr<Module> module) {
    module->field_name_ = name;
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
    // methods.
    for (const auto& m : class_compilation_unit()->get_functions()) {
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
    // find_offset() method reads "dict_" object.
    // Lock because another thread can modify "dict_" object at the same time
    // calling insert() method.
    // Ideally recursive_mutex should be replaced with shared_mutex (C++ 17)
    // for the performance reasons.
    std::unique_lock<std::recursive_mutex> keeper(create_method_guard_);
    auto offset = find_offset(name, EntityType::METHOD);
    if (offset) {
      return methods_[*offset].get();
    }

    if (const std::shared_ptr<Function>& fn =
            class_compilation_unit()->find_function(name)) {
      // lock because technically this is marked const,
      // but we have to update the internal Method cache.
      // This can be removed when class_compilation_unit() is the source of
      // truth for methods.
      Module* mutable_this = const_cast<Module*>(this);
      std::unique_ptr<Method> m(new Method(mutable_this, fn));
      return mutable_this
          ->insert(
              fn->name(),
              mutable_this->methods_,
              EntityType::METHOD,
              std::move(m))
          .get();
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
  void train(bool on = true);
  /// Calls train(false) to enable "eval" mode.
  /// Do not override this method, override `train()` instead.
  void eval() {
    train(/*on=*/false);
  }
  /// True if the module is in training mode.
  bool is_training() {
    if (auto p = find_attribute("training")) {
      return p->value().toBool();
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
  void to(at::Device device, at::ScalarType dtype, bool non_blocking = false);

  /// Recursively casts all parameters to the given dtype.
  ///
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  void to(at::ScalarType dtype, bool non_blocking = false);

  /// Recursively moves all parameters to the given device.
  ///
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  void to(at::Device device, bool non_blocking = false);

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
      const ModuleLookup& module_lookup,
      // translate current module singleton type to new module
      // singleton type.
      std::unordered_map<TypePtr, TypePtr>& type_remap,
      std::vector<std::string> names = {}) const;

  void clone_method(
      const Module& orig,
      const std::string& name,
      const std::unordered_map<TypePtr, TypePtr>& type_remap);

  void clone_method(const Module& orig, const std::string& name);

  enum class EntityType { MODULE, PARAMETER, ATTRIBUTE, METHOD };

  at::optional<EntityType> kind_of(const std::string& name) const {
    auto it = dict_.find(name);
    if (it == dict_.end()) {
      // methods are lazily created, see if this is, in face,
      // a method that has not been created yet.
      if (auto fn = class_compilation_unit()->find_function(name)) {
        return EntityType::METHOD;
      }
      return at::nullopt;
    }
    return it->second.type;
  }

  ModulePtr module_object() const {
    return module_value_;
  }
  std::shared_ptr<CompilationUnit> class_compilation_unit() {
    return module_object()->type()->compilation_unit();
  }
  std::shared_ptr<const CompilationUnit> class_compilation_unit() const {
    return module_object()->type()->compilation_unit();
  }

  // so that C++ users can easily add methods
  void define(const std::string& src, const ResolverPtr& resolver = nullptr);

  template <typename... Types>
  IValue create_class(const c10::QualifiedName& name, Types&&... args) const {
    return create_class(name, {IValue(std::forward<Types>(args))...});
  }

  IValue create_class(const c10::QualifiedName& name, Stack stack) const;

 private:
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

  // modules have a single namespace, but spread over 4 different concepts:
  // parameters, attributes, methods, and sub-modules
  // we store individual lists of each concept, and a single map to
  // unify the namespace and ensure fast lookup
  std::vector<std::shared_ptr<Module>> modules_;
  std::vector<Slot> parameters_;
  std::vector<Slot> attributes_;
  std::vector<std::unique_ptr<Method>> methods_;

  std::unordered_map<std::string, Entry> dict_;

  // The name of the module as it appears as a submodule
  // parent.myself = this_module
  // 'myself' is the field name:
  std::string field_name_;

  ModulePtr module_value_;

  // Currently we are in a transitionary state
  // where we construct such first class functions but we lower them
  // to a form where the modules does not exist before execution.

  // So each Method is actually stored twice once in first-class Module
  // form and once in lowered form.

  // first-class: module_value_->type().compilation_unit() holds Functions that
  // treat modules as first class.

  mutable std::recursive_mutex create_method_guard_;
  friend struct Method;
};

TORCH_API bool& getInlineEverythingMode();

} // namespace script
} // namespace jit
} // namespace torch
