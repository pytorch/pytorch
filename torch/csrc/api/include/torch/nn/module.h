#pragma once

#include <torch/nn/pimpl.h>
#include <torch/ordered_dict.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <ATen/ATen.h>

#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <type_traits>

namespace torch {
namespace nn {

/// The base class for all modules in PyTorch.
///
/// \rst
/// .. note::
///   The design and implementation of this class is largely based on the Python
///   API. You may want to consult the python documentation for
///   :py:class:`pytorch:torch.nn.Module` for further clarification on certain
///   methods or behavior.
/// \endrst
///
/// A `Module` is an abstraction over the implementation of some function or
/// algorithm, possibly associated with some persistent data. A `Module` may
/// contain further `Module`s ("submodules"), each with their own
/// implementation, persistent data and further submodules. `Module`s can thus
/// be said to form a recursive tree structure. A `Module` is registered as a
/// submodule to another `Module` by calling `register_module()`, typically from
/// within a parent module's constructor.
///
/// A distinction is made between three kinds of persistent data that may be
/// associated with a `Module`:
///
/// 1. *Parameters*: tensors that record gradients, typically weights updated
///    during the backward step (e.g. the `weight` of a `Linear` module),
/// 2. *Buffers*: tensors that do not record gradients, typically updated during
///    the forward step, such as running statistics (e.g. `mean` and `variance`
///    in the `BatchNorm` module),
/// 3. Any additional state, not necessarily tensors, required for the
///    implementation or configuration of a `Module`.
///
/// The first two kinds of state are special in that they may be registered
/// with the `Module` system to allow convenient access and batch configuration.
/// For example, registered parameters in any `Module` may be iterated over via
/// the `parameters()` accessor. Further, changing the data type of a `Module`'s
/// registered parameters can be done conveniently via `Module::to()`, e.g.
/// `module->to(torch::kCUDA)` to move all parameters to GPU memory. Lastly,
/// registered parameters and buffers are handled specially during a `clone()`
/// operation, which performs a deepcopy of a cloneable `Module` hierarchy.
///
/// Parameters are registered with a `Module` via `register_parameter`. Buffers
/// are registered separately via `register_buffer`. These methods are part of
/// the public API of `Module` and are typically invoked from within a
/// concrete `Module`s constructor.
class TORCH_API Module : public std::enable_shared_from_this<Module> {
 public:
  using ModuleApplyFunction = std::function<void(Module&)>;
  using ConstModuleApplyFunction = std::function<void(const Module&)>;
  using NamedModuleApplyFunction =
      std::function<void(const std::string&, Module&)>;
  using ConstNamedModuleApplyFunction =
      std::function<void(const std::string&, const Module&)>;
  using ModulePointerApplyFunction =
      std::function<void(const std::shared_ptr<Module>&)>;
  using NamedModulePointerApplyFunction =
      std::function<void(const std::string&, const std::shared_ptr<Module>&)>;

  /// Tells the base `Module` about the name of the submodule.
  explicit Module(std::string name);

  /// Constructs the module without immediate knowledge of the submodule's name.
  /// The name of the submodule is inferred via RTTI (if possible) the first
  /// time `.name()` is invoked.
  Module();

  virtual ~Module() = default;

  /// Returns the name of the `Module`.
  ///
  /// A `Module` has an associated `name`, which is a string representation of
  /// the kind of concrete `Module` it represents, such as `"Linear"` for the
  /// `Linear` module. Under most circumstances, this name is automatically
  /// inferred via runtime type information (RTTI). In the unusual circumstance
  /// that you have this feature disabled, you may want to manually name your
  /// `Module`s by passing the string name to the `Module` base class'
  /// constructor.
  const std::string& name() const noexcept;

  /// Performs a recursive deep copy of the module and all its registered
  /// parameters, buffers and submodules.
  ///
  /// Optionally, this method sets the current device
  /// to the one supplied before cloning. If no device is given, each
  /// parameter and buffer will be moved to the device of its source.
  ///
  /// \rst
  /// .. attention::
  ///   Attempting to call the `clone()` method inherited from the base `Module`
  ///   class (the one documented here) will fail. To inherit an actual
  ///   implementation of `clone()`, you must subclass `Cloneable`. `Cloneable`
  ///   is templatized on the concrete module type, and can thus properly copy a
  ///   `Module`. This method is provided on the base class' API solely for an
  ///   easier-to-use polymorphic interface.
  /// \endrst
  virtual std::shared_ptr<Module> clone(
      const optional<Device>& device = nullopt) const;

  /// Applies the `function` to the `Module` and recursively to every submodule.
  /// The function must accept a `Module&`.
  ///
  /// \rst
  /// .. code-block:: cpp
  ///   MyModule module;
  ///   module->apply([](nn::Module& module) {
  ///     std::cout << module.name() << std::endl;
  ///   });
  /// \endrst
  void apply(const ModuleApplyFunction& function);

  /// Applies the `function` to the `Module` and recursively to every submodule.
  /// The function must accept a `const Module&`.
  ///
  /// \rst
  /// .. code-block:: cpp
  ///   MyModule module;
  ///   module->apply([](const nn::Module& module) {
  ///     std::cout << module.name() << std::endl;
  ///   });
  /// \endrst
  void apply(const ConstModuleApplyFunction& function) const;

  /// Applies the `function` to the `Module` and recursively to every submodule.
  /// The function must accept a `const std::string&` for the key of the module,
  /// and a `Module&`. The key of the module itself is the empty string. If
  /// `name_prefix` is given, it is prepended to every key as
  /// `<name_prefix>.<key>` (and just `name_prefix` for the module itself).
  ///
  /// \rst
  /// .. code-block:: cpp
  ///   MyModule module;
  ///   module->apply([](const std::string& key, nn::Module& module) {
  ///     std::cout << key << ": " << module.name() << std::endl;
  ///   });
  /// \endrst
  void apply(
      const NamedModuleApplyFunction& function,
      const std::string& name_prefix = std::string());

  /// Applies the `function` to the `Module` and recursively to every submodule.
  /// The function must accept a `const std::string&` for the key of the module,
  /// and a `const Module&`. The key of the module itself is the empty string.
  /// If `name_prefix` is given, it is prepended to every key as
  /// `<name_prefix>.<key>` (and just `name_prefix` for the module itself).
  ///
  /// \rst
  /// .. code-block:: cpp
  ///   MyModule module;
  ///   module->apply([](const std::string& key, const nn::Module& module) {
  ///     std::cout << key << ": " << module.name() << std::endl;
  ///   });
  /// \endrst
  void apply(
      const ConstNamedModuleApplyFunction& function,
      const std::string& name_prefix = std::string()) const;

  /// Applies the `function` to the `Module` and recursively to every submodule.
  /// The function must accept a `const std::shared_ptr<Module>&`.
  ///
  /// \rst
  /// .. code-block:: cpp
  ///   MyModule module;
  ///   module->apply([](const std::shared_ptr<nn::Module>& module) {
  ///     std::cout << module->name() << std::endl;
  ///   });
  /// \endrst
  void apply(const ModulePointerApplyFunction& function) const;

  /// Applies the `function` to the `Module` and recursively to every submodule.
  /// The function must accept a `const std::string&` for the key of the module,
  /// and a `const std::shared_ptr<Module>&`. The key of the module itself is
  /// the empty string. If `name_prefix` is given, it is prepended to every key
  /// as
  /// `<name_prefix>.<key>` (and just `name_prefix` for the module itself).
  ///
  /// \rst
  /// .. code-block:: cpp
  ///   MyModule module;
  ///   module->apply([](const std::string& key,
  ///                    const std::shared_ptr<nn::Module>& module) {
  ///     std::cout << key << ": " << module->name() << std::endl;
  ///   });
  /// \endrst
  void apply(
      const NamedModulePointerApplyFunction& function,
      const std::string& name_prefix = std::string()) const;

  /// Returns the parameters of this `Module` and if `recurse` is true, also
  /// recursively of every submodule.
  std::vector<Tensor> parameters(bool recurse = true) const;

  /// Returns an `OrderedDict` with the parameters of this `Module` along with
  /// their keys, and if `recurse` is true also recursively of every submodule.
  OrderedDict<std::string, Tensor> named_parameters(bool recurse = true) const;

  /// Returns the buffers of this `Module` and if `recurse` is true, also
  /// recursively of every submodule.
  std::vector<Tensor> buffers(bool recurse = true) const;

  /// Returns an `OrderedDict` with the buffers of this `Module` along with
  /// their keys, and if `recurse` is true also recursively of every submodule.
  OrderedDict<std::string, Tensor> named_buffers(bool recurse = true) const;

  /// Returns the submodules of this `Module` (the entire submodule hierarchy)
  /// and if `include_self` is true, also inserts a `shared_ptr` to this module
  /// in the first position.
  ///
  /// \rst
  /// .. warning::
  ///   Only pass `include_self` as `true` if this `Module` is stored in a
  ///   `shared_ptr`! Otherwise an exception will be thrown. You may still call
  ///   this method with `include_self` set to false if your `Module` is not
  ///   stored in a `shared_ptr`.
  /// \endrst
  std::vector<std::shared_ptr<Module>> modules(bool include_self = true) const;

  /// Returns an `OrderedDict` of the submodules of this `Module` (the entire
  /// submodule hierarchy) and their keys, and if `include_self` is true, also
  /// inserts a `shared_ptr` to this module in the first position. If
  /// `name_prefix` is given, it is prepended to every key as
  /// `<name_prefix>.<key>` (and just `name_prefix` for the module itself).
  ///
  /// \rst
  /// .. warning::
  ///   Only pass `include_self` as `true` if this `Module` is stored in a
  ///   `shared_ptr`! Otherwise an exception will be thrown. You may still call
  ///   this method with `include_self` set to false if your `Module` is not
  ///   stored in a `shared_ptr`.
  /// \endrst
  OrderedDict<std::string, std::shared_ptr<Module>> named_modules(
      const std::string& name_prefix = std::string(),
      bool include_self = true) const;

  /// Returns the direct submodules of this `Module`.
  std::vector<std::shared_ptr<Module>> children() const;

  /// Returns an `OrderedDict` of the direct submodules of this `Module` and
  /// their keys.
  OrderedDict<std::string, std::shared_ptr<Module>> named_children() const;

  /// Enables "training" mode.
  virtual void train(bool on = true);

  /// Calls train(false) to enable "eval" mode.
  /// Do not override this method, override `train()` instead.
  void eval();

  /// True if the module is in training mode.
  ///
  /// Every `Module` has a boolean associated with it that determines whether
  /// the `Module` is currently in *training* mode (set via `.train()`) or in
  /// *evaluation* (inference) mode (set via `.eval()`). This property is
  /// exposed via `is_training()`, and may be used by the implementation of a
  /// concrete module to modify its runtime behavior. See the `BatchNorm` or
  /// `Dropout` modules for examples of `Module`s that use different code paths
  /// depending on this property.
  virtual bool is_training() const noexcept;

  /// Recursively casts all parameters to the given `dtype` and `device`.
  ///
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  virtual void to(
      torch::Device device,
      torch::Dtype dtype,
      bool non_blocking = false);

  /// Recursively casts all parameters to the given dtype.
  ///
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  virtual void to(torch::Dtype dtype, bool non_blocking = false);

  /// Recursively moves all parameters to the given device.
  ///
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  virtual void to(torch::Device device, bool non_blocking = false);

  /// Recursively zeros out the `grad` value of each registered parameter.
  virtual void zero_grad();

  /// Attempts to cast this `Module` to the given `ModuleType`.
  ///
  /// This method is useful when calling `apply()`.
  /// \rst
  /// .. code-block:: cpp
  ///
  ///   void initialize_weights(nn::Module& module) {
  ///     torch::NoGradGuard no_grad;
  ///     if (auto* linear = module.as<nn::Linear>()) {
  ///       linear->weight.normal_(0.0, 0.02);
  ///     }
  ///   }
  ///
  ///   MyModule module;
  ///   module->apply(initialize_weights);
  /// \endrst
  template <typename ModuleType>
  typename ModuleType::ContainedType* as() noexcept;

  /// Attempts to cast this `Module` to the given `ModuleType`.
  ///
  /// This method is useful when calling `apply()`.
  /// \rst
  /// .. code-block:: cpp
  ///   void initialize_weights(nn::Module& module) {
  ///     torch::NoGradGuard no_grad;
  ///     if (auto* linear = module.as<nn::Linear>()) {
  ///       linear->weight.normal_(0.0, 0.02);
  ///     }
  ///   }
  ///
  ///   MyModule module;
  ///   module->apply(initialize_weights);
  /// \endrst
  template <typename ModuleType>
  const typename ModuleType::ContainedType* as() const noexcept;

  /// Attempts to cast this `Module` to the given `ModuleType`.
  ///
  /// This method is useful when calling `apply()`.
  /// \rst
  /// .. code-block:: cpp
  ///
  ///   void initialize_weights(nn::Module& module) {
  ///     torch::NoGradGuard no_grad;
  ///     if (auto* linear = module.as<nn::Linear>()) {
  ///       linear->weight.normal_(0.0, 0.02);
  ///     }
  ///   }
  ///
  ///   MyModule module;
  ///   module.apply(initialize_weights);
  /// \endrst
  template <
      typename ModuleType,
      typename = torch::detail::disable_if_module_holder_t<ModuleType>>
  ModuleType* as() noexcept;

  /// Attempts to cast this `Module` to the given `ModuleType`.
  ///
  /// This method is useful when calling `apply()`.
  /// \rst
  /// .. code-block:: cpp
  ///
  ///   void initialize_weights(nn::Module& module) {
  ///     torch::NoGradGuard no_grad;
  ///     if (auto* linear = module.as<nn::Linear>()) {
  ///       linear->weight.normal_(0.0, 0.02);
  ///     }
  ///   }
  ///
  ///   MyModule module;
  ///   module.apply(initialize_weights);
  /// \endrst
  template <
      typename ModuleType,
      typename = torch::detail::disable_if_module_holder_t<ModuleType>>
  const ModuleType* as() const noexcept;

  /// Serializes the `Module` into the given `OutputArchive`.
  ///
  /// If the `Module` contains unserializable submodules (e.g. `nn::Functional`),
  /// those submodules are skipped when serializing.
  virtual void save(serialize::OutputArchive& archive) const;

  /// Deserializes the `Module` from the given `InputArchive`.
  ///
  /// If the `Module` contains unserializable submodules (e.g. `nn::Functional`),
  /// we don't check the existence of those submodules in the `InputArchive` when
  /// deserializing.
  virtual void load(serialize::InputArchive& archive);

  /// Streams a pretty representation of the `Module` into the given `stream`.
  /// By default, this representation will be the name of the module (taken from
  /// `name()`), followed by a recursive pretty print of all of the `Module`'s
  /// submodules.
  ///
  /// Override this method to change the pretty print. The input
  /// `stream` should be returned from the method, to allow easy chaining.
  virtual void pretty_print(std::ostream& stream) const;

  /// Returns whether the `Module` is serializable.
  virtual bool is_serializable() const;

  /// Registers a parameter with this `Module`.
  ///
  /// A parameter should be any gradient-recording tensor used in the
  /// implementation of your `Module`. Registering it makes it available to
  /// methods such as `parameters()`, `clone()` or `to().`
  ///
  /// Note that registering an undefined Tensor (e.g. `module.register_parameter("param", Tensor())`)
  /// is allowed, and is equivalent to `module.register_parameter("param", None)` in Python API.
  ///
  /// \rst
  /// .. code-block:: cpp
  ///
  ///   MyModule::MyModule() {
  ///     weight_ = register_parameter("weight", torch::randn({A, B}));
  ///   }
  /// \endrst
  Tensor& register_parameter(
      std::string name,
      Tensor tensor,
      bool requires_grad = true);

  /// Registers a buffer with this `Module`.
  ///
  /// A buffer is intended to be state in your module that does not record
  /// gradients, such as running statistics. Registering it makes it available
  /// to methods such as `buffers()`, `clone()` or `to().
  ///
  /// \rst
  /// .. code-block:: cpp
  ///
  ///   MyModule::MyModule() {
  ///     mean_ = register_buffer("mean", torch::empty({num_features_}));
  ///   }
  /// \endrst
  Tensor& register_buffer(std::string name, Tensor tensor);

  /// Registers a submodule with this `Module`.
  ///
  /// Registering a module makes it available to methods such as `modules()`,
  /// `clone()` or `to()`.
  ///
  /// \rst
  /// .. code-block:: cpp
  ///
  ///   MyModule::MyModule() {
  ///     submodule_ = register_module("linear", torch::nn::Linear(3, 4));
  ///   }
  /// \endrst
  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      std::string name,
      std::shared_ptr<ModuleType> module);

  /// Registers a submodule with this `Module`.
  ///
  /// This method deals with `ModuleHolder`s.
  ///
  /// Registering a module makes it available to methods such as `modules()`,
  /// `clone()` or `to()`.
  ///
  /// \rst
  /// .. code-block:: cpp
  ///
  ///   MyModule::MyModule() {
  ///     submodule_ = register_module("linear", torch::nn::Linear(3, 4));
  ///   }
  /// \endrst
  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      std::string name,
      ModuleHolder<ModuleType> module_holder);

  /// Replaces a registered submodule with this `Module`.
  ///
  /// This takes care of the registration, if you used submodule members, you should
  //  assign the submodule as well, i.e. use as
  ///     module->submodule_ = module->replace_module("linear", torch::nn::Linear(3, 4));
  /// It only works when a module of the name is already registered.
  ///
  /// This is useful for replacing a module after initialization, e.g.
  /// for finetuning.
  template <typename ModuleType>
  std::shared_ptr<ModuleType> replace_module(
      const std::string& name,
      std::shared_ptr<ModuleType> module);

  /// Replaces a registered submodule with this `Module`.
  /// This method deals with `ModuleHolder`s.
  ///
  /// This takes care of the registration, if you used submodule members, you should
  //  assign the submodule as well, i.e. use as
  ///     module->submodule_ = module->replace_module("linear", linear_holder);
  /// It only works when a module of the name is already registered.
  ///
  /// This is useful for replacing a module after initialization, e.g.
  /// for finetuning.
  template <typename ModuleType>
  std::shared_ptr<ModuleType> replace_module(
      const std::string& name,
      ModuleHolder<ModuleType> module_holder);

  /// Unregisters a submodule from this `Module`. If there is no such module
  /// with `name` an exception is thrown.
  void unregister_module(const std::string& name);

 private:
  // Friend classes.

  template <typename Derived>
  friend class Cloneable;

  /// Pretty prints the given `Module` into the `ostream`.
  TORCH_API friend std::ostream& operator<<(
      std::ostream& stream,
      const nn::Module& module);

  // data parallel using this method to configure gradient edges during the
  // replicate step.
  template <typename ModuleType>
  friend void replicate_grad_edges(
      const std::shared_ptr<Module>& module,
      const std::vector<std::shared_ptr<ModuleType>>& replicas,
      const std::vector<Device>& devices);

  // Private methods.

  /// Used in the implementation of `Cloneable`.
  virtual void clone_(Module& other, const optional<Device>& device);

  /// The implementation of the various `to()` methods.
  template <typename... Ts>
  void to_impl(Ts&&... ts);

  /// Implements pretty printing the module hierarchy.
  void pretty_print_recursive(
      std::ostream& stream,
      const std::string& indentation) const;

  /// Applies the `function` to every submodule recursively, starting at this
  /// `Module`'s children (thus not including the module itself).
  void apply_to_submodules(
      const NamedModulePointerApplyFunction& function,
      const std::string& name_prefix = std::string()) const;

  /// Returns a shared_ptr to `this` in a safe (checked) way.
  std::shared_ptr<Module> shared_from_this_checked() const;

  /// The registered parameters of this `Module`.
  OrderedDict<std::string, Tensor> parameters_;

  /// The registered buffers of this `Module`.
  OrderedDict<std::string, Tensor> buffers_;

  /// The registered (direct) submodules of this `Module`.
  OrderedDict<std::string, std::shared_ptr<Module>> children_;

  /// The module's name (e.g. "LSTM").
  mutable optional<std::string> name_;

  /// Whether the module is in training mode.
  bool is_training_{true};
};

/// Serialize a `Module` pointer into an `OutputArchive`.
TORCH_API serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const std::shared_ptr<nn::Module>& module);

/// Deserializes a `Module` from an `InputArchive`.
TORCH_API serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    const std::shared_ptr<nn::Module>& module);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ nn::Module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename ModuleType>
typename ModuleType::ContainedType* Module::as() noexcept {
  // Use the contained type of the `ModuleHolder`, e.g. `LinearImpl` for
  // `Linear`, since `LinearImpl` inherits `nn::Module`.
  return as<typename ModuleType::ContainedType>();
}

template <typename ModuleType>
const typename ModuleType::ContainedType* Module::as() const noexcept {
  // Use the contained type of the `ModuleHolder`, e.g. `LinearImpl` for
  // `Linear`, since `LinearImpl` inherits `nn::Module`.
  return as<typename ModuleType::ContainedType>();
}

template <typename ModuleType, typename>
ModuleType* Module::as() noexcept {
  return dynamic_cast<ModuleType*>(this);
}

template <typename ModuleType, typename>
const ModuleType* Module::as() const noexcept {
  return dynamic_cast<const ModuleType*>(this);
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::register_module(
    std::string name,
    std::shared_ptr<ModuleType> module) {
  TORCH_CHECK(!name.empty(), "Submodule name must not be empty");
  TORCH_CHECK(
      name.find('.') == std::string::npos,
      "Submodule name must not contain a dot (got '",
      name,
      "')");
  auto& base_module = children_.insert(std::move(name), std::move(module));
  return std::dynamic_pointer_cast<ModuleType>(base_module);
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::register_module(
    std::string name,
    ModuleHolder<ModuleType> module_holder) {
  return register_module(std::move(name), module_holder.ptr());
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::replace_module(
    const std::string& name,
    std::shared_ptr<ModuleType> module) {
  auto& base_module = (children_[name] = std::move(module));
  return std::dynamic_pointer_cast<ModuleType>(base_module);
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::replace_module(
    const std::string& name,
    ModuleHolder<ModuleType> module_holder) {
  return replace_module(name, module_holder.ptr());
}

template <typename... Ts>
void Module::to_impl(Ts&&... ts) {
  // First call `to()` on every child module.
  for (auto& child : children_) {
    child.value()->to(ts...);
  }
  // Then move every parameter to the new dtype/device.
  for (auto& parameter : named_parameters(/*recurse=*/false)) {
    parameter->set_data(autograd::Variable(*parameter).to(ts...));
  }
  // Then move every buffer to the new dtype/device.
  for (auto& buffer : named_buffers(/*recurse=*/false)) {
    buffer->set_data(autograd::Variable(*buffer).to(ts...));
  }
}

} // namespace nn
} // namespace torch
