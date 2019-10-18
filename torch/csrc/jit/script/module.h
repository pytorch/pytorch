#pragma once
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/named_value.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
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

struct Module;

template <typename T>
struct slot_list_impl;

struct NameModule;
struct NameValue;

using module_list = slot_list_impl<NameModule>;
using ivalue_list = slot_list_impl<NameValue>;
using ModuleLookup = std::function<Module(const std::vector<std::string>&)>;

enum class EntityType { MODULE, PARAMETER, ATTRIBUTE, METHOD };

// A method in a module, e.g. f in:
//
// class M(ScriptModule):
//   @script_method
//   def f(self, x):
//     ...
// Note: because Method/Module are exposed to python these
// classes use python method naming conventions
struct TORCH_API Method {
  Method(ModulePtr owner, Function* function);

  // the module that contains this method.
  Module owner() const;
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
  ModulePtr owner_;

  // Underlying unbound function
  // This is the _lowered_ function and is different than the
  // first-class function in class_compilation_unit()
  Function* function_;
};

struct TORCH_API Module {
  explicit Module(c10::QualifiedName class_name);
  Module(std::shared_ptr<CompilationUnit> cu, const c10::ClassTypePtr& type);
  Module(
      c10::QualifiedName,
      std::shared_ptr<CompilationUnit> cu,
      bool shouldMangle = false);
  // module_value_ null and will be lazily initialized if is needed
  Module() {}
  Module(ModulePtr module_value) : module_value_(std::move(module_value)) {}
  ~Module() {}

  const c10::QualifiedName& name() const {
    return *module_object()->type()->name();
  }

  void set_optimized(bool o) {
    AT_WARN(
        "Module::set_optimized() is deprecated and has no effect. "
        "Please use setGraphExecutorOptimize()");
  }

  bool is_optimized() const {
    AT_WARN(
        "Module::is_optimized() is deprecated and always returns true. "
        "Please use getGraphExecutorOptimize()");
    return true;
  }

  IValue forward(std::vector<IValue> inputs) {
    return get_method("forward")(std::move(inputs));
  }

  // In script modules, buffers are Tensors attribute that are _not_ registered
  // as parameters. This is different than in nn.Module where there is a special
  // register_buffer method. With this simplification, we only need to track
  // whether a slot is a parameter to be able to classify it.
  void register_buffer(const std::string& name, autograd::Variable v) {
    type()->addOrCheckAttribute(name, TensorType::get());
    module_object()->setAttr(name, v);
  }
  void register_parameter(
      const std::string& name,
      autograd::Variable v,
      bool is_buffer) {
    type()->addOrCheckAttribute(name, TensorType::get(), !is_buffer);
    module_object()->setAttr(name, v);
  }
  void register_attribute(
      const std::string& name,
      const TypePtr t,
      IValue v,
      bool is_param = false) {
    type()->addOrCheckAttribute(name, t, is_param);
    module_object()->setAttr(name, v);
  }
  void register_module(const std::string& name, const Module& module) {
    type()->addOrCheckAttribute(name, module.type());
    module_object()->setAttr(name, module.module_object());
  }

  void set_parameter(const std::string& name, at::Tensor v) {
    module_object()->setAttr(name, v);
  }

  autograd::Variable get_parameter(const std::string& name) const {
    return autograd::as_variable_ref(module_object()->getAttr(name).toTensor());
  }

  IValue get_attribute(const std::string& name) const {
    return module_object()->getAttr(name);
  }

  void set_attribute(const std::string& name, IValue v) const {
    return module_object()->setAttr(name, v);
  }

  autograd::Variable get_buffer(const std::string& name) const {
    return autograd::as_variable_ref(get_attribute(name).toTensor());
  }

  // each module owns its method. The reference returned here
  // is guarenteed to stay valid until this module has been destroyed
  Method get_method(const std::string& name) const {
    if (auto method = find_method(name)) {
      return *method;
    }
    AT_ERROR("Method '", name, "' is not defined.");
  }

  Module get_module(const std::string& name) const {
    auto obj = module_object()->getAttr(name).toObject();
    return Module(obj);
  }

  ivalue_list get_slots() const;

  module_list get_modules() const;

  ivalue_list get_parameters() const;

  ivalue_list get_attributes() const;

  void dump(
      bool print_method_bodies,
      bool print_attr_values,
      bool print_param_values) const;

  std::string dump_to_str(
      bool print_method_bodies,
      bool print_attr_values,
      bool print_param_values,
      int level) const;

  const std::vector<Method> get_methods() const {
    return fmap(
        type()->methods(),
        [&](Function* func) {
          return Method(module_object(), func);
        });
  }

  c10::optional<autograd::Variable> find_parameter(
      const std::string& name) const;
  c10::optional<IValue> find_attribute(const std::string& name) const;
  c10::optional<autograd::Variable> find_buffer(const std::string& name) const;
  c10::optional<Module> find_module(const std::string& name) const;
  c10::optional<Method> find_method(const std::string& basename) const;

  void apply(const std::function<void(Module&)>& fn);

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
      return p->toBool();
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
      const ExtraFilesMap& extra_files = ExtraFilesMap()) const;

  void save(
      const std::string& filename,
      const ExtraFilesMap& extra_files = ExtraFilesMap()) const;

  void _save_for_mobile(
      std::ostream& out,
      const ExtraFilesMap& extra_files = ExtraFilesMap()) const;

  void _save_for_mobile(
      const std::string& filename,
      const ExtraFilesMap& extra_files = ExtraFilesMap()) const;

  // Create a deep copy of this module.
  Module clone() const;

  void clone_method(const Module& orig, const std::string& name);

  ModulePtr module_object() const;

  ClassTypePtr type() const {
    return module_object()->type();
  }
  std::shared_ptr<CompilationUnit> class_compilation_unit() const {
    return module_object()->compilation_unit();
  }

  // so that C++ users can easily add methods
  void define(const std::string& src, const ResolverPtr& resolver = nullptr);

  template <typename... Types>
  IValue create_class(const c10::QualifiedName& name, Types&&... args) const {
    return create_class(name, {IValue(std::forward<Types>(args))...});
  }

  IValue create_class(const c10::QualifiedName& name, Stack stack) const;

  size_t num_slots() const {
    return module_object()->slots().size();
  }
  c10::optional<EntityType> entity_type(const std::string& name) const {
    if (auto slot_idx = type()->findAttributeSlot(name)) {
      return entity_type(*slot_idx);
    }
    return c10::nullopt;
  }
  EntityType entity_type(size_t offset_) const {
    TORCH_CHECK(offset_ < type()->numAttributes());
    if (type()->is_parameter(offset_)) {
      return EntityType::PARAMETER;
    }
    at::TypePtr t = type()->getAttribute(offset_);
    if (auto cls = t->cast<at::ClassType>()) {
      if (cls->is_module()) {
        return EntityType::MODULE;
      }
    }
    return EntityType::ATTRIBUTE;
  }

 private:
  Module clone_impl(std::unordered_map<TypePtr, TypePtr>& type_remap) const;

  void clone_method(
      const Module& orig,
      const Function& method,
      const std::unordered_map<TypePtr, TypePtr>& type_remap);

  c10::QualifiedName getNameForMethod(std::string basename) const {
    return QualifiedName(name(), basename);
  }

  void to_impl(
      const c10::optional<at::Device>& device,
      const c10::optional<at::ScalarType>& dtype,
      bool non_blocking);

  // mutable be we lazily initialize in module_object.
  mutable ModulePtr module_value_;
};

struct NameModule {
  std::string name;
  Module module;
};

struct NameValue {
  std::string name;
  IValue value;
};

// this iterator for the slot list defined below has a position in the list i_
// and an optional field type_ that if present
// restricts iteration to only the slots of module_ that
// have EntityType *type_. This allows it to return, e.g.
// only the parameter slots.
// The template parameter allows us to use the same implementation for a list
// that returns Module via template specialization of the operator* method.
template <typename T>
struct TORCH_API slot_iterator_impl {
  slot_iterator_impl(Module module, c10::optional<EntityType> type, size_t i)
      : module_(module), type_(type), i_(i) {
    advance_to_valid();
  }
  T operator*() const;
  T operator->() const {
    return **this;
  }
  slot_iterator_impl& operator++() {
    ++i_;
    advance_to_valid();
    return *this;
  }
  slot_iterator_impl operator++(int) {
    slot_iterator_impl old = *this;
    ++(*this);
    return old;
  }

 private:
  void advance_to_valid() {
    while (i_ < module_.num_slots() &&
           (type_ && module_.entity_type(i_) != *type_)) {
      ++i_;
    }
  }
  Module module_;
  c10::optional<EntityType> type_;
  size_t i_;

  template <typename TT>
  friend inline bool operator!=(
      const slot_iterator_impl<TT>& a,
      const slot_iterator_impl<TT>& b);
};

template <>
inline NameModule slot_iterator_impl<NameModule>::operator*() const {
  return {module_.type()->getAttributeName(i_),
          module_.module_object()->getSlot(i_).toObject()};
}

template <>
inline NameValue slot_iterator_impl<NameValue>::operator*() const {
  return {module_.type()->getAttributeName(i_),
          module_.module_object()->getSlot(i_)};
}

template <typename T>
inline bool operator!=(
    const slot_iterator_impl<T>& a,
    const slot_iterator_impl<T>& b) {
  return a.i_ != b.i_;
}

// This type represents lists of parameters, attributes, and
// submodules contained in the module. It is abstract because
// they are not stored directly in std::vectors but inside the
// module's IValue object itself.
template <typename T>
struct TORCH_API slot_list_impl {
  using iterator = slot_iterator_impl<T>;
  using const_iterator = slot_iterator_impl<T>;
  slot_iterator_impl<T> begin() const {
    return slot_iterator_impl<T>(module_, type_, 0);
  }
  slot_iterator_impl<T> end() const {
    return slot_iterator_impl<T>(module_, type_, module_.num_slots());
  }
  size_t size() const {
    if (!size_) {
      size_ = size_t(0);
      for (T s : *(this)) {
        ++*size_;
      }
    }
    return *size_;
  }

 private:
  slot_list_impl(Module module, c10::optional<EntityType> type)
      : module_(std::move(module)), type_(type) {
    if (!type_) {
      size_ = module_.num_slots();
    }
  }
  Module module_;
  // only include Slots of the following type
  c10::optional<EntityType> type_;
  // size of this list, cached on first request
  // when we need to filter the slot list
  mutable c10::optional<size_t> size_;
  friend struct Module;
};

TORCH_API bool& getInlineEverythingMode();

} // namespace script
} // namespace jit
} // namespace torch
