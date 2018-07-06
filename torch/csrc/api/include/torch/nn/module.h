#pragma once

#include <torch/detail/ordered_dict.h>
#include <torch/nn/cursor.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>
#include <ATen/optional.h>

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
  /// parameters, buffers and submodules.
  virtual std::shared_ptr<Module> clone() const;

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

  template <class Archive>
  void save(Archive& ar) const {
    auto params = parameters();
    size_t size = params.size();
    ar(size);
    for (auto& p : params) {
      ar(p.key, p.value);
    }
  }

  template <class Archive>
  void load(Archive& ar) {
    auto params = parameters();
    size_t size;
    ar(size);
    std::string name;
    for (size_t i = 0; i < size; i++) {
      ar(name);
      ar(params[name]);
    }
  }

  /// Returns true if the dynamic type of this module is of the given
  /// `ModuleType`. Performs a `dynamic_cast` to check this.
  template <
      typename ModuleType,
      typename = torch::detail::disable_if_module_holder_t<ModuleType>>
  bool is() const noexcept {
    return dynamic_cast<const ModuleType*>(this) != nullptr;
  }

  /// Returns true if the dynamic type of this module is of the given
  /// `ModuleType`. Performs a `dynamic_cast` to check this.
  template <typename ModuleType>
  torch::enable_if_t<torch::detail::is_module_holder<ModuleType>::value, bool>
  is() const noexcept {
    // Use the contained type of the `ModuleHolder`, e.g. `LinearImpl` for
    // `Linear`, since `LinearImpl` inherits `nn::Module`.
    return is<typename ModuleType::ContainedType>();
  }

 protected:
  Tensor& register_parameter(
      std::string name,
      Tensor tensor,
      bool requires_grad = true);
  Tensor& register_buffer(std::string name, Tensor tensor);

  template <
      typename ModuleType,
      typename = torch::detail::disable_if_module_holder_t<ModuleType>>
  std::shared_ptr<ModuleType> register_module(
      std::string name,
      std::shared_ptr<ModuleType> module) {
    auto& base_module = children_.insert(std::move(name), std::move(module));
    return std::static_pointer_cast<ModuleType>(base_module);
  }

  template <typename ModuleHolderType>
  ModuleHolderType register_module(
      std::string name,
      ModuleHolderType module_holder) {
    register_module(std::move(name), module_holder.get());
    return module_holder;
  }

 private:
  template <typename T>
  using OrderedDict = torch::detail::OrderedDict<std::string, T>;

  template <typename Derived>
  friend class Cloneable;
  template <typename T>
  friend class detail::CursorBase;

  virtual void clone_(Module& other);

  /// The implementation of the various `to()`.
  template <typename... Ts>
  void to_impl(Ts&&... ts) {
    // First call `to()` on every child module.
    for (auto& child : children_) {
      child.value->to(ts...);
    }
    // Then move every parameter to the new dtype/device.
    for (auto& parameter : parameters_) {
      at::detail::set_data(*parameter, parameter->data().to(ts...));
    }
    // Then move every buffer to the new dtype/device.
    for (auto& buffer : buffers_) {
      at::detail::set_data(*buffer, buffer->data().to(ts...));
    }
  }

  OrderedDict<Tensor> parameters_;
  OrderedDict<Tensor> buffers_;
  OrderedDict<std::shared_ptr<Module>> children_;

  /// The module's name (e.g. "LSTM").
  mutable at::optional<std::string> name_;

  /// Whether the module is in training mode.
  bool is_training_{true};
};
} // namespace nn
} // namespace torch
