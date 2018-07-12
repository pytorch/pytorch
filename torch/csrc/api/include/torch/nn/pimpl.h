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
using disable_if_module_holder_t = disable_if_t<is_module_holder<T>::value>;
} // namespace detail

namespace nn {

/// A `ModuleHolder` is essentially a wrapper around `std::shared_ptr<M>` where
/// `M` is an `nn::Module` subclass, with convenient constructors defined for
/// the kind of constructions we want to allow for our modules.
template <typename Contained>
class ModuleHolder : torch::detail::ModuleHolderIndicator {
 protected:
  /// The module pointer this class wraps.
  /// NOTE: Must be placed at the top of the class so that we can use it with
  /// trailing return types below.
  std::shared_ptr<Contained> impl_;

 public:
  using ContainedType = Contained;

  /// Constructs the `ModuleHolder` with an empty contained value. Access to
  /// the underlying module is not permitted and will throw an exception, until
  /// a value is assigned.
  explicit ModuleHolder(std::nullptr_t) : impl_(nullptr) {}

  /// Constructs the `ModuleHolder` with a contained module, forwarding all
  /// arguments to its constructor.
  template <typename... Ts>
  explicit ModuleHolder(Ts&&... ts)
      : impl_(new Contained(std::forward<Ts>(ts)...)) {}

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
    return get();
  }

  /// Forwards to the contained module.
  const Contained* operator->() const {
    return get();
  }

  /// Returns a reference to the contained module.
  Contained& operator*() {
    return *get();
  }

  /// Returns a const reference to the contained module.
  const Contained& operator*() const {
    return *get();
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

  /// Returns a const pointer to the underlying module.
  const Contained* get() const {
    AT_CHECK(!is_empty(), "Accessing empty ModuleHolder");
    return impl_.get();
  }

  /// Forwards to the call operator of the contained module.
  template <typename... Args>
  auto operator()(Args&&... args)
      -> decltype((*impl_)(std::forward<Args>(args)...)) {
    return (*impl_)(std::forward<Args>(args)...);
  }

  /// Forwards to the subscript operator of the contained module.
  template <typename Arg>
  auto operator[](Arg&& arg) -> decltype((*impl_)[std::forward<Arg>(arg)]) {
    return (*impl_)[std::forward<Arg>(arg)];
  }

  /// Returns true if the `ModuleHolder` does not contain a module.
  bool is_empty() const noexcept {
    return impl_ == nullptr;
  }
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
#define TORCH_MODULE_IMPL(Name, Impl)                            \
  class Name : public torch::nn::ModuleHolder<Impl> {            \
   public:                                                       \
    using torch::nn::ModuleHolder<Impl>::ModuleHolder;           \
    Name(const Name&) = default;                                 \
    Name(Name&&) = default;                                      \
    Name(Name& other) : Name(static_cast<const Name&>(other)) {} \
    Name& operator=(const Name&) = default;                      \
    Name& operator=(Name&&) = default;                           \
  }

/// Like `TORCH_MODULE_IMPL`, but defaults the `Impl` name to `<Name>Impl`.
#define TORCH_MODULE(Name) TORCH_MODULE_IMPL(Name, Name##Impl)
