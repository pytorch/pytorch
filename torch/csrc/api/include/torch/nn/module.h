#pragma once

#include <torch/detail/ordered_dict.h>
#include <torch/nn/cursor.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>
#include <ATen/core/optional.h>

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace torch {
namespace detail {
template <typename T>
class CursorBase;
} // namespace detail
} // namespace torch

namespace torch {
namespace nn {

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
  const std::string& name() const noexcept;

  /// Performs a recursive deep copy of the module and all its registered
  /// parameters, buffers and submodules, optionally setting the current device
  /// to the one supplied before cloning. If no device is given, each
  /// parameter and buffer will be moved to the device of its source.
  virtual std::shared_ptr<Module> clone(
      at::optional<Device> device = at::nullopt) const;

  /// Provides a means to traverse the `Module` tree.
  ModuleCursor modules();
  ConstModuleCursor modules() const;

  /// Traverses the (immediate) children of the `Module`.
  ModuleCursor children();
  ConstModuleCursor children() const;

  /// Provides a means to recursively access the parameters of the `Module`
  /// tree.
  ParameterCursor parameters();
  ConstParameterCursor parameters() const;

  /// Provides a means to recursively access the buffers of the `Module` tree.
  BufferCursor buffers();
  ConstBufferCursor buffers() const;

  /// Enables training mode.
  virtual void train();

  /// Disables training mode.
  virtual void eval();

  /// True if the module is in training mode.
  virtual bool is_training() const noexcept;

  /// Recursively casts all parameters to the given dtype and device.
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  virtual void to(
      torch::Device device,
      torch::Dtype dtype,
      bool non_blocking = false);

  /// Recursively casts all parameters to the given dtype.
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  virtual void to(torch::Dtype dtype, bool non_blocking = false);

  /// Recursively moves all parameters to the given device.
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  virtual void to(torch::Device device, bool non_blocking = false);

  /// Recursively zeros out the `grad` values of all parameters.
  virtual void zero_grad();

  /// Serializes the `Module`.
  template <class Archive>
  void save(Archive& ar) const;

  /// Deserializes the `Module`.
  template <class Archive>
  void load(Archive& ar);

  /// Attempts to cast this `Module` to the given `ModuleType`.
  template <typename ModuleType>
  typename ModuleType::ContainedType* as() noexcept;

  /// Attempts to cast this `Module` to the given `ModuleType`.
  template <
      typename ModuleType,
      typename = torch::detail::disable_if_module_holder_t<ModuleType>>
  ModuleType* as() noexcept;

 protected:
  /// Registers a parameter with this `Module`.
  Tensor& register_parameter(
      std::string name,
      Tensor tensor,
      bool requires_grad = true);
  /// Registers a buffer with this `Module`.
  Tensor& register_buffer(std::string name, Tensor tensor);

  /// Registers a submodule with this `Module`.
  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      std::string name,
      std::shared_ptr<ModuleType> module);

  /// Registers a submodule with this `Module`.
  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      std::string name,
      ModuleHolder<ModuleType> module_holder);

 private:
  template <typename T>
  using OrderedDict = torch::detail::OrderedDict<std::string, T>;

  template <typename Derived>
  friend class Cloneable;
  template <typename T>
  friend class detail::CursorBase;

  virtual void clone_(Module& other, at::optional<Device> device);

  /// The implementation of the various `to()` methods.
  template <typename... Ts>
  void to_impl(Ts&&... ts);

  OrderedDict<Tensor> parameters_;
  OrderedDict<Tensor> buffers_;
  OrderedDict<std::shared_ptr<Module>> children_;

  /// The module's name (e.g. "LSTM").
  mutable at::optional<std::string> name_;

  /// Whether the module is in training mode.
  bool is_training_{true};
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ nn::Module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <class Archive>
void Module::save(Archive& ar) const {
  auto params = parameters();
  size_t size = params.size();
  ar(size);
  for (auto& p : params) {
    ar(p.key, p.value);
  }
}

template <class Archive>
void Module::load(Archive& ar) {
  auto params = parameters();
  size_t size;
  ar(size);
  std::string name;
  for (size_t i = 0; i < size; i++) {
    ar(name);
    ar(params[name]);
  }
}

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
    at::detail::set_data(
        *parameter, autograd::Variable(*parameter).data().to(ts...));
  }
  // Then move every buffer to the new dtype/device.
  for (auto& buffer : buffers_) {
    at::detail::set_data(*buffer, autograd::Variable(*buffer).data().to(ts...));
  }
}

} // namespace nn
} // namespace torch
