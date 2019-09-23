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

template <typename T>
struct slot_list_impl;
using slot_list = slot_list_impl<Slot>;
using module_list = slot_list_impl<Module>;
using ModuleLookup = std::function<Module(const std::vector<std::string>&)>;

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
  Module(std::shared_ptr<CompilationUnit> cu, c10::ClassTypePtr type);
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
    set_or_add_slot(name, TensorType::get(), v, EntityType::ATTRIBUTE);
  }

  void register_parameter(
      const std::string& name,
      autograd::Variable v,
      bool is_buffer) {
    set_or_add_slot(
        name,
        TensorType::get(),
        v,
        is_buffer ? EntityType::ATTRIBUTE : EntityType::PARAMETER);
  }
  void register_attribute(
      const std::string& name,
      const TypePtr type,
      IValue ivalue) {
    set_or_add_slot(name, type, ivalue, EntityType::ATTRIBUTE);
  }
  void register_module(const std::string& name, const Module& module) {
    set_or_add_slot(
        name, module.type(), module.module_object(), EntityType::MODULE);
  }

  void set_parameter(const std::string& name, at::Tensor v) {
    get_slot(name, EntityType::PARAMETER).setValue(v);
  }

  autograd::Variable get_parameter(const std::string& name) const {
    return autograd::as_variable_ref(
        get_slot(name, EntityType::PARAMETER).value().toTensor());
  }

  IValue get_attribute(const std::string& name) const {
    return get_slot(name, EntityType::ATTRIBUTE).value();
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
    auto obj = get_slot(name, EntityType::MODULE).value().toObject();
    return Module(obj);
  }

  module_list get_modules() const;
  slot_list get_slots() const;
  slot_list get_parameters() const;
  slot_list get_attributes() const;
  slot_list get_module_slots() const;

  void dump(
      bool print_method_bodies,
      bool print_attr_values,
      bool print_param_values) const;

  const std::vector<Method> get_methods() const {
    return fmap(
        type()->methods(),
        [&](Function* func) {
          return Method(module_object(), func);
        });
  }

  c10::optional<Slot> find_parameter(const std::string& name) const {
    return find_slot(name, EntityType::PARAMETER);
  }
  c10::optional<Slot> find_attribute(const std::string& name) {
    return find_slot(name, EntityType::ATTRIBUTE);
  }
  c10::optional<Slot> find_buffer(const std::string& name) {
    auto iv = find_attribute(name);
    if (iv && iv->type()->isSubtypeOf(TensorType::get())) {
      return iv;
    }
    return c10::nullopt;
  }
  c10::optional<Module> find_module(const std::string& name) const {
    if (auto slot = find_slot(name, EntityType::MODULE)) {
      return Module(slot->value().toObject());
    }
    return c10::nullopt;
  }
  c10::optional<Method> find_method(const std::string& basename) const {
    for (Function* fn : type()->methods()) {
      if (fn->name() == basename) {
        return Method(module_object(), fn);
      }

    }
    return c10::nullopt;
  }
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
      const ExtraFilesMap& extra_files = ExtraFilesMap()) const;

  void save(
      const std::string& filename,
      const ExtraFilesMap& extra_files = ExtraFilesMap()) const;

  // Create a deep copy of this module.
  Module clone() const;

  void clone_method(const Module& orig, const std::string& name);

  at::optional<EntityType> kind_of(const std::string& name) const {
    if (find_method(name)) {
      return EntityType::METHOD;
    }
    if (auto offset = type()->findAttributeSlot(name)) {
      return get_slot(*offset).entity_type();
    }
    return c10::nullopt;
  }

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

  Slot get_slot(size_t slot) const {
    TORCH_CHECK(
        slot < module_object()->slots().size(), "not a valid slot offset");
    return Slot(module_object(), slot);
  }

  size_t num_slots() const {
    return module_object()->slots().size();
  }

  void _finalize() {
    isInitializing_ = false;
  }

 private:
  // We need a way to represent a module in the following state:
  //   The type is fully initialized, but the module state is not.
  // In particular, slot resolution doesn't work in this case.
  //
  // TODO: figure out a better way
  bool isInitializing_ = false;

  Module clone_impl(std::unordered_map<TypePtr, TypePtr>& type_remap) const;

  std::string _dump_to_string(
      bool omit_method_bodies,
      bool omit_attr_values,
      bool omit_param_values,
      int level) const;

  void clone_method(
      const Module& orig,
      const Function& method,
      const std::unordered_map<TypePtr, TypePtr>& type_remap);

  c10::QualifiedName getNameForMethod(std::string basename) const {
    return QualifiedName(name(), basename);
  }
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
  void check_entity(EntityType expected, size_t slot) const {
    EntityType actual = get_slot(slot).entity_type();
    TORCH_CHECK(
        expected == actual,
        "The field '",
        type()->getAttributeName(slot),
        "' is a ",
        toString(actual),
        " but this call is"
        " trying to use it as a ",
        toString(expected));
  }

  void set_or_add_slot(
      const std::string& name,
      const TypePtr& slot_type,
      IValue v,
      EntityType etype) {
    auto slot = type()->findAttributeSlot(name);
    if (!slot) {
      slot =
          type()->addAttribute(name, slot_type, etype == EntityType::PARAMETER);
    } else {
      if (!isInitializing_) {
        // Skip this check if the module is in an initializing state, as the
        // slot may not actually be filled.
        check_entity(etype, *slot);
      }
    }
    TypePtr atype = type()->getAttribute(*slot);
    TORCH_CHECK(slot_type->isSubtypeOf(atype));
    module_object()->setSlot(*slot, std::move(v));
  }

  Slot get_slot(const std::string& name, EntityType etype) const {
    size_t slot = type()->getAttributeSlot(name);
    check_entity(etype, slot);
    return get_slot(slot);
  }
  c10::optional<Slot> find_slot(const std::string& name, EntityType etype)
      const {
    auto slot = type()->findAttributeSlot(name);
    if (!slot) {
      return c10::nullopt;
    }
    Slot r = get_slot(*slot);
    if (r.entity_type() != etype) {
      return c10::nullopt;
    }
    return r;
  }

  void to_impl(
      const c10::optional<at::Device>& device,
      const c10::optional<at::ScalarType>& dtype,
      bool non_blocking);

  // mutable be we lazily initialize in module_object.
  mutable ModulePtr module_value_;
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
           (type_ && module_.get_slot(i_).entity_type() != *type_)) {
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
inline Slot slot_iterator_impl<Slot>::operator*() const {
  return module_.get_slot(i_);
}

template <>
inline Module slot_iterator_impl<Module>::operator*() const {
  return Module(module_.get_slot(i_).to_module());
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
