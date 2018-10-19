#pragma once

#include <torch/detail/ordered_dict.h>
#include <torch/nn/cursor.h>
#include <torch/nn/pimpl.h>
#include <torch/serialize/archive.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>
#include "c10/util/Optional.h"

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

// forward declarations confuse doxygen
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace torch {
namespace detail {
template <typename T>
class CursorBase;
} // namespace detail
} // namespace torch
#endif // DOXYGEN_SHOULD_SKIP_THIS

namespace torch {
namespace nn {

/// The base class for all modules in PyTorch.
///
/// \rst
/// .. note::
///   The design and implementation of this class is largely based on the Python
///   API. You may want to consult [its
///   documentation](https://pytorch.org/docs/master/nn.html#torch.nn.Module)
///   for further clarification on certain methods or behavior.
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
/// the protected API of `Module` and are typically invoked from within a
/// concrete `Module`s constructor.
class Module {
 public:
  /// Tells the base `Module` about the name of the submodule.
  explicit Module(std::string name);

  /// Constructs the base module without immediate knowledge of the submodule's
  /// name. The name of the submodule is inferred via RTTI the first time
  /// `.name()` is invoked.
  Module() = default;

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
      c10::optional<Device> device = c10::nullopt) const;

  /// Provides a means to traverse the `Module` tree.
  ///
  /// See the documentation for `CursorBase` for information on how to operate
  /// on the returned cursor.
  ModuleCursor modules();

  /// Provides a means to traverse the `Module` tree.
  ///
  /// See the documentation for `CursorBase` for information on how to operate
  /// on the returned cursor.
  ConstModuleCursor modules() const;

  /// Traverses the (immediate) children of the `Module`.
  ///
  /// See the documentation for `CursorBase` for information on how to operate
  /// on the returned cursor.
  ModuleCursor children();

  /// Traverses the (immediate) children of the `Module`.
  ///
  /// See the documentation for `CursorBase` for information on how to operate
  /// on the returned cursor.
  ConstModuleCursor children() const;

  /// Provides a means to recursively access the parameters of the `Module`
  /// tree.
  ///
  /// See the documentation for `CursorBase` for information on how to operate
  /// on the returned cursor.
  ParameterCursor parameters();

  /// Provides a means to recursively access the parameters of the `Module`
  /// tree.
  ///
  /// See the documentation for `CursorBase` for information on how to operate
  /// on the returned cursor.
  ConstParameterCursor parameters() const;

  /// Provides a means to recursively access the buffers of the `Module` tree.
  ///
  /// See the documentation for `CursorBase` for information on how to operate
  /// on the returned cursor.
  BufferCursor buffers();

  /// Provides a means to recursively access the buffers of the `Module` tree.
  ///
  /// See the documentation for `CursorBase` for information on how to operate
  /// on the returned cursor.
  ConstBufferCursor buffers() const;

  /// Enables training mode.
  virtual void train();

  /// Disables training mode.
  virtual void eval();

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
  /// This method is useful when calling `apply()` on a `ModuleCursor`.
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
  ///   module->modules().apply(initialize_weights);
  /// \endrst
  template <typename ModuleType>
  typename ModuleType::ContainedType* as() noexcept;

  /// Attempts to cast this `Module` to the given `ModuleType`.
  ///
  /// This method is useful when calling `apply()` on a `ModuleCursor`.
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
  ///   module->modules().apply(initialize_weights);
  /// \endrst
  template <
      typename ModuleType,
      typename = torch::detail::disable_if_module_holder_t<ModuleType>>
  ModuleType* as() noexcept;

  /// Serializes the `Module` into the given `OutputArchive`.
  virtual void save(serialize::OutputArchive& archive) const;

  /// Deserializes the `Module` from the given `InputArchive`.
  virtual void load(serialize::InputArchive& archive);

 protected:
  /// Registers a parameter with this `Module`.
  ///
  /// A parameter should be any gradient-recording tensor used in the
  /// implementation of your `Module`. Registering it makes it available to
  /// methods such as `parameters()`, `clone()` or `to().`
  ///
  /// \rst
  /// .. code-block:: cpp
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
  ///   MyModule::MyModule() {
  ///     submodule_ = register_module("linear", torch::nn::Linear(3, 4));
  ///   }
  /// \endrst
  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      std::string name,
      ModuleHolder<ModuleType> module_holder);

 private:
  template <typename T>
  using OrderedDict = torch::detail::OrderedDict<std::string, T>;

  // Friend classes.

  template <typename Derived>
  friend class Cloneable;
  template <typename T>
  friend class detail::CursorBase;

  // Private methods.

  /// Used in the implementation of `Cloneable`.
  virtual void clone_(Module& other, c10::optional<Device> device);

  /// The implementation of the various `to()` methods.
  template <typename... Ts>
  void to_impl(Ts&&... ts);

  /// The registered parameters of this `Module`.
  OrderedDict<Tensor> parameters_;

  /// The registered buffers of this `Module`.
  OrderedDict<Tensor> buffers_;

  /// The registered (direct) submodules of this `Module`.
  OrderedDict<std::shared_ptr<Module>> children_;

  /// The module's name (e.g. "LSTM").
  mutable c10::optional<std::string> name_;

  /// Whether the module is in training mode.
  bool is_training_{true};
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ nn::Module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename ModuleType>
typename ModuleType::ContainedType* Module::as() noexcept {
  // Use the contained type of the `ModuleHolder`, e.g. `LinearImpl` for
  // `Linear`, since `LinearImpl` inherits `nn::Module`.
  return as<typename ModuleType::ContainedType>();
}

template <typename ModuleType, typename>
ModuleType* Module::as() noexcept {
  return dynamic_cast<ModuleType*>(this);
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::register_module(
    std::string name,
    std::shared_ptr<ModuleType> module) {
  auto& base_module = children_.insert(std::move(name), std::move(module));
  return std::dynamic_pointer_cast<ModuleType>(base_module);
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::register_module(
    std::string name,
    ModuleHolder<ModuleType> module_holder) {
  return register_module(std::move(name), module_holder.ptr());
}

template <typename... Ts>
void Module::to_impl(Ts&&... ts) {
  // First call `to()` on every child module.
  for (auto& child : children_) {
    child.value->to(ts...);
  }
  // Then move every parameter to the new dtype/device.
  for (auto& parameter : parameters_) {
    parameter->set_data(autograd::Variable(*parameter).data().to(ts...));
  }
  // Then move every buffer to the new dtype/device.
  for (auto& buffer : buffers_) {
    buffer->set_data(autograd::Variable(*buffer).data().to(ts...));
  }
}

} // namespace nn
} // namespace torch
