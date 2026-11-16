#pragma once

#include <torch/nn/modules/container/any.h>
#include <torch/types.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace torch::nn {

/// Stores a type erased `Module` with name.
///
/// The `NamedAnyModule` class enables the following API for constructing
/// `nn::Sequential` with named submodules:
/// \rst
/// .. code-block:: cpp
///
///   struct M : torch::nn::Module {
///     explicit M(int value_) : value(value_) {}
///     int value;
///     int forward() {
///       return value;
///     }
///   };
///
///   Sequential sequential({
///     {"m1", std::make_shared<M>(1)},  // shared pointer to `Module` is
///     supported {std::string("m2"), M(2)},  // `Module` is supported
///     {"linear1", Linear(10, 3)}  // `ModuleHolder` is supported
///   });
/// \endrst
class NamedAnyModule {
 public:
  /// Creates a `NamedAnyModule` from a (boxed) `Module`.
  template <typename ModuleType>
  NamedAnyModule(std::string name, std::shared_ptr<ModuleType> module_ptr)
      : NamedAnyModule(std::move(name), AnyModule(std::move(module_ptr))) {}

  /// Creates a `NamedAnyModule` from a `Module`, moving or copying it
  /// into a `shared_ptr` internally.
  // NOTE: We need to use `std::remove_reference_t<M>` to get rid of
  // any reference components for make_unique.
  template <typename M, typename = torch::detail::enable_if_module_t<M>>
  NamedAnyModule(std::string name, M&& module)
      : NamedAnyModule(
            std::move(name),
            std::make_shared<std::remove_reference_t<M>>(
                std::forward<M>(module))) {}

  /// Creates a `NamedAnyModule` from a `Module` that is unwrapped from
  /// a `ModuleHolder`.
  template <typename M>
  NamedAnyModule(std::string name, const ModuleHolder<M>& module_holder)
      : NamedAnyModule(std::move(name), module_holder.ptr()) {}

  /// Creates a `NamedAnyModule` from a type-erased `AnyModule`.
  NamedAnyModule(std::string name, AnyModule any_module)
      : name_(std::move(name)), module_(std::move(any_module)) {}

  /// Returns a reference to the name.
  const std::string& name() const noexcept {
    return name_;
  }

  /// Returns a reference to the module.
  AnyModule& module() noexcept {
    return module_;
  }

  /// Returns a const reference to the module.
  const AnyModule& module() const noexcept {
    return module_;
  }

 private:
  std::string name_;
  AnyModule module_;
};

} // namespace torch::nn
