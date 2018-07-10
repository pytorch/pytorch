#pragma once

#include <torch/csrc/utils/variadic.h>
#include <torch/tensor.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace torch {
namespace detail {
/// This class exists  only to do SFINAE on abstract types `T` that are really
/// `ModuleHolder<ModuleType>`, because there's no good way to say that `T` is a
/// `ModuleHolder` over some unknown type `ModuleType`. With this, you can do
/// enable_if_t<is_base_of<ModuleHolderIndicator, T>::value>::type.
struct ModuleHolderIndicator {};

template <typename T>
using is_module_holder = std::is_base_of<ModuleHolderIndicator, decay_t<T>>;

template <typename T>
using disable_if_module_holder_t =
    disable_if_t<std::is_base_of<ModuleHolderIndicator, decay_t<T>>::value>;
} // namespace detail

namespace nn {

/// A `ModuleHolder` is essentially a wrapper around `std::shared_ptr<M>` where
/// `M` is an `nn::Module` subclass, with convenient constructors defined for
/// the kind of constructions we want to allow for our modules.
template <typename Contained>
class ModuleHolder : torch::detail::ModuleHolderIndicator {
 public:
  using ContainedType = Contained;

  /// Constructs the `ModuleHolder` with an empty contained value.
  ModuleHolder() = default;

  /// Single argument constructor of the underlying type.
  /// Example: `Linear(4)` or `Linear(LinearOptions(4))`.
  template <typename T>
  explicit ModuleHolder(T&& t)
      : impl_(std::make_shared<Contained>(std::forward<T>(t))) {}

  /// Multi-argument constructor. This constructor is special in that the
  /// expectation is that the constructor of the contained type takes an object
  /// that can be constructed with the given arguments. For our modules, this is
  /// always the `Options` struct. For this reason, the arguments are forwarded
  /// inside braces, as to construct the constructor argument.
  /// Example: `Linear(3, 4)`, equivalent to `Linear(LinearOptions(3, 4))`.
  template <typename T, typename... Ts>
  explicit ModuleHolder(T&& t, Ts&&... ts)
      : impl_(new Contained({std::forward<T>(t), std::forward<Ts>(ts)...})) {}

  /// Constructs the `ModuleHolder` from a pointer to the contained type.
  /// Example: `Linear(std::make_shared<LinearImpl>(...))`.
  /* implicit */ ModuleHolder(std::shared_ptr<Contained> module)
      : impl_(std::move(module)) {}

  /// Returns true if the `ModuleHolder` contains a module, or false if it is
  /// `nullptr`.
  explicit operator bool() const noexcept {
    return !is_empty();
  }

  /// Forwards to the contained module.
  Contained* operator->() {
    AT_CHECK(!is_empty(), "Accessing empty ModuleHolder");
    return impl_.get();
  }

  /// Forwards to the contained module.
  const Contained* operator->() const {
    AT_CHECK(!is_empty(), "Accessing empty ModuleHolder");
    return impl_.get();
  }

  /// Forwards to the call operator of the contained module.
  template <typename... Args>
  Tensor operator()(Args&&... args) {
    return (*impl_)(std::forward<Args>(args)...);
  }

  /// Returns a shared pointer to the underlying module.
  const std::shared_ptr<Contained>& ptr() const {
    AT_CHECK(!is_empty(), "Accessing empty ModuleHolder");
    return impl_;
  }

  /// Returns a pointer to the underlying module.
  Contained* get() {
    AT_CHECK(!is_empty(), "Accessing empty ModuleHolder");
    return impl_.get();
  }

  /// Returns a pointer to the underlying module.
  const Contained* get() const {
    AT_CHECK(!is_empty(), "Accessing empty ModuleHolder");
    return impl_.get();
  }

  /// Returns true if the `ModuleHolder` does not contain a module.
  bool is_empty() const noexcept {
    return impl_ == nullptr;
  }

 protected:
  /// The module pointer this class wraps.
  std::shared_ptr<Contained> impl_;
};
} // namespace nn
} // namespace torch

#define TORCH_ARG(T, name)                          \
  auto name(const T& new_##name)->decltype(*this) { \
    this->name##_ = new_##name;                     \
    return *this;                                   \
  }                                                 \
  auto name(T&& new_##name)->decltype(*this) {      \
    this->name##_ = std::move(new_##name);          \
    return *this;                                   \
  }                                                 \
  const T& name() const noexcept {                  \
    return this->name##_;                           \
  }                                                 \
  T name##_

/// Defines a class `Name` which inherits from `nn::ModuleHolder` to provide a
/// wrapper over a `std::shared_ptr<Impl>`.
#define TORCH_MODULE_IMPL(Name, Impl)                  \
  class Name : public torch::nn::ModuleHolder<Impl> {  \
   public:                                             \
    using torch::nn::ModuleHolder<Impl>::ModuleHolder; \
  }

/// Like `TORCH_MODULE_IMPL`, but defaults the `Impl` name to `<Name>Impl`.
#define TORCH_MODULE(Name) TORCH_MODULE_IMPL(Name, Name##Impl)
