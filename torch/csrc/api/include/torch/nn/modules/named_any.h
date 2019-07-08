#pragma once

#include <torch/detail/static.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/any.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/memory.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/Device.h>

#include <initializer_list>
#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

/// Stores a type erased `Module` with name.
///
/// The `NamedAnyModule` class and the `modules_ordered_dict(...)` function enables
/// the following API for constructing `nn::Sequential` with named submodules:
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
///   Sequential sequential(modules_ordered_dict({
///     {"m1", std::make_shared<M>(1)},  // shared pointer to `Module` is supported
///     {std::string("m2"), M(2)},  // `Module` is supported
///     {"linear1", Linear(10, 3)}  // `ModuleHolder` is supported
///   }));
/// \endrst
///
/// Specifically, we design the signature of `modules_ordered_dict(...)` to be
/// `modules_ordered_dict(std::initializer_list<NamedAnyModule> named_modules)`, as
/// a result of evaluating the following possible approaches:
///
/// Approach 1:
/// `modules_ordered_dict(std::initializer_list<
///   torch::OrderedDict<std::string, ModuleType>::Item> named_modules)`
///
/// Why it doens't work:
/// When we pass in a braced-init list such as
/// `modules_ordered_dict({{"m1", M(1)}, {"m2", M(2)}})`, at the template argument
/// deduction step the compiler is not able to deduce the type of `ModuleType` to
/// the type of `M(1)` or `M(2)`, since the compiler doesn't actually look into the
/// braced-init list `{"m1", M(1)}` and figure out what the types of its elements are.
///
/// Approach 2:
/// `modules_ordered_dict(std::initializer_list<
///   std::pair<std::string, AnyModule> named_modules)`
///
/// Why it doens't work:
/// When we pass in a braced-init list such as
/// `modules_ordered_dict({{"m1", M(1)}, {"m2", M(2)}})`, the compiler is not able to
/// match `std::initializer_list<std::pair<std::string, AnyModule>>` to the nested
/// braced-init list `{{"m1", M(1)}, {"m2", M(2)}}`, and results in a "could not
/// convert" error.
///
/// Approach 3:
/// `modules_ordered_dict(std::initializer_list<NamedAnyModule> named_modules)`
///
/// Why it works:
/// When we pass in a braced-init list such as
/// `modules_ordered_dict({{"m1", M(1)}, {"m2", M(2)}})`, the compiler is passing the
/// braced-init lists {"m1", M(1)} and {"m2", M(2)} to the `NamedAnyModule`
/// constructors, and the constructors are able to figure out the types of the
/// braced-init lists' elements and match to the correct module type.

class NamedAnyModule {
 public:
  /// Creates a `NamedAnyModule` from a (boxed) `Module`.
  template <typename ModuleType>
  NamedAnyModule(std::string name, std::shared_ptr<ModuleType> module_ptr)
      : NamedAnyModule(std::move(name), AnyModule(std::move(module_ptr))) {}

  /// Creates a `NamedAnyModule` from a `Module`, moving or copying it
  /// into a `shared_ptr` internally.
  // NOTE: We need to use `std::remove_reference<M>::type` to get rid of
  // any reference components for make_unique.
  template <typename M, typename = torch::detail::enable_if_module_t<M>>
  NamedAnyModule(std::string name, M&& module)
      : NamedAnyModule(
          std::move(name),
          std::make_shared<typename std::remove_reference<M>::type>(
            std::forward<M>(module))) {}

  /// Creates a `NamedAnyModule` from a `Module` that is unwrapped from
  /// a `ModuleHolder`.
  template <typename M>
  NamedAnyModule(std::string name, const ModuleHolder<M>& module_holder)
      : NamedAnyModule(std::move(name), module_holder.ptr()) {}

  /// Returns a reference to the name.
  const std::string& name() const noexcept {
    return name_;
  }

  /// Returns a reference to the module.
  AnyModule& module() noexcept {
    return module_;
  }

 private:
  /// Creates a `NamedAnyModule` from a type-erased `AnyModule`.
  NamedAnyModule(std::string name, AnyModule any_module)
    : name_(std::move(name)), module_(std::move(any_module)) {}

  std::string name_;
  AnyModule module_;
};

TORCH_API torch::OrderedDict<std::string, AnyModule> modules_ordered_dict(
  std::initializer_list<NamedAnyModule> named_modules);

} // namespace nn
} // namespace torch
