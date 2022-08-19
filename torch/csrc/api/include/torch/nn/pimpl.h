#pragma once

#include <torch/arg.h>
#include <torch/detail/static.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <torch/csrc/utils/variadic.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace torch {
namespace detail {
// Dump all the template metaprogramming in this file.
#include <torch/csrc/api/include/torch/nn/pimpl-inl.h>
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
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<Contained> impl_;

 public:
  using ContainedType = Contained;

  /// Default constructs the contained module if if has a default constructor,
  /// else produces a static error.
  ///
  /// NOTE: This uses the behavior of template
  /// classes in C++ that constructors (or any methods) are only compiled when
  /// actually used.
  ModuleHolder() : impl_(default_construct()) {
    static_assert(
        std::is_default_constructible<Contained>::value,
        "You are trying to default construct a module which has "
        "no default constructor. Use = nullptr to give it the empty state "
        "(e.g. `Linear linear = nullptr;` instead of `Linear linear;`).");
  }

  /// Constructs the `ModuleHolder` with an empty contained value. Access to
  /// the underlying module is not permitted and will throw an exception, until
  /// a value is assigned.
  /* implicit */ ModuleHolder(std::nullptr_t) : impl_(nullptr) {}

  /// Constructs the `ModuleHolder` with a contained module, forwarding all
  /// arguments to its constructor.
  template <
      typename Head,
      typename... Tail,
      typename = typename std::enable_if<
          !(torch::detail::is_module_holder_of<Head, ContainedType>::value &&
            (sizeof...(Tail) == 0))>::type>
  explicit ModuleHolder(Head&& head, Tail&&... tail)
      : impl_(new Contained(
            std::forward<Head>(head),
            std::forward<Tail>(tail)...)) {}

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
    TORCH_CHECK(!is_empty(), "Accessing empty ModuleHolder");
    return impl_;
  }

  /// Returns a pointer to the underlying module.
  Contained* get() {
    TORCH_CHECK(!is_empty(), "Accessing empty ModuleHolder");
    return impl_.get();
  }

  /// Returns a const pointer to the underlying module.
  const Contained* get() const {
    TORCH_CHECK(!is_empty(), "Accessing empty ModuleHolder");
    return impl_.get();
  }

  /// Calls the `forward()` method of the contained module.
  template <typename... Args>
  auto operator()(Args&&... args)
      -> torch::detail::return_type_of_forward_t<Contained, Args...> {
    // This will not compile if the module does not have a `forward()` method
    // (as expected).
    // NOTE: `std::forward` is qualified to prevent VS2017 emitting
    // error C2872: 'std': ambiguous symbol
    return impl_->forward(::std::forward<Args>(args)...);
  }

  /// Forwards to the subscript operator of the contained module.
  /// NOTE: std::forward is qualified to prevent VS2017 emitting
  ///       error C2872: 'std': ambiguous symbol
  template <typename Arg>
  decltype(auto) operator[](Arg&& arg) {
    return (*impl_)[::std::forward<Arg>(arg)];
  }

  /// Returns true if the `ModuleHolder` does not contain a module.
  bool is_empty() const noexcept {
    return impl_ == nullptr;
  }

 private:
  /// In C++17, the two methods below could be written as the following:
  /// if constexpr (std::is_default_constructible_v<Contained>) {
  ///   return std::make_shared<Contained>();
  /// } else {
  ///   return nullptr;
  /// }
  /// In C++11, we use SFINAE instead of `if constexpr`.

  template <
      typename T = Contained,
      typename = torch::enable_if_t<std::is_default_constructible<T>::value>>
  std::shared_ptr<Contained> default_construct() {
    return std::make_shared<Contained>();
  }

  template <typename T = Contained>
  torch::disable_if_t<
      std::is_default_constructible<T>::value,
      std::shared_ptr<Contained>>
  default_construct() {
    return nullptr;
  }
};

/// Pretty prints the given `Module` into the `ostream`.
template <typename ModuleType>
std::ostream& operator<<(
    std::ostream& stream,
    const nn::ModuleHolder<ModuleType>& module) {
  return stream << *module;
}

/// Serializes a `ModuleHolder` into an `OutputArchive`.
template <typename ModuleType>
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const nn::ModuleHolder<ModuleType>& module) {
  return archive << module.ptr();
}

/// Deserializes a `ModuleHolder` from an `InputArchive`.
template <typename ModuleType>
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    nn::ModuleHolder<ModuleType>& module) {
  return archive >> module.ptr();
}

} // namespace nn
} // namespace torch

/// Defines a class `Name` which inherits from `nn::ModuleHolder` to provide a
/// wrapper over a `std::shared_ptr<ImplType>`.
/// `Impl` is a type alias for `ImplType` which provides a way to call static
/// method of `ImplType`.
#define TORCH_MODULE_IMPL(Name, ImplType)                              \
  class Name : public torch::nn::ModuleHolder<ImplType> { /* NOLINT */ \
   public:                                                             \
    using torch::nn::ModuleHolder<ImplType>::ModuleHolder;             \
    using Impl C10_UNUSED = ImplType;                                  \
  }

/// Like `TORCH_MODULE_IMPL`, but defaults the `ImplType` name to `<Name>Impl`.
#define TORCH_MODULE(Name) TORCH_MODULE_IMPL(Name, Name##Impl)
