#pragma once

#include <torch/detail/static.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/container/any_module_holder.h>
#include <torch/nn/modules/container/any_value.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/memory.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/Device.h>

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

/// Stores a type erased `Module`.
///
/// The PyTorch C++ API does not impose an interface on the signature of
/// `forward()` in `Module` subclasses. This gives you complete freedom to
/// design your `forward()` methods to your liking. However, this also means
/// there is no unified base type you could store in order to call `forward()`
/// polymorphically for any module. This is where the `AnyModule` comes in.
/// Instead of inheritance, it relies on type erasure for polymorphism.
///
/// An `AnyModule` can store any `nn::Module` subclass that provides a
/// `forward()` method. This `forward()` may accept any types and return any
/// type. Once stored in an `AnyModule`, you can invoke the underlying module's
/// `forward()` by calling `AnyModule::forward()` with the arguments you would
/// supply to the stored module (though see one important limitation below).
/// Example:
///
/// \rst
/// .. code-block:: cpp
///
///   struct GenericTrainer {
///     torch::nn::AnyModule module;
///
///     void train(torch::Tensor input) {
///       module.forward(input);
///     }
///   };
///
///   GenericTrainer trainer1{torch::nn::Linear(3, 4)};
///   GenericTrainer trainer2{torch::nn::Conv2d(3, 4, 2)};
/// \endrst
///
/// As `AnyModule` erases the static type of the stored module (and its
/// `forward()` method) to achieve polymorphism, type checking of arguments is
/// moved to runtime. That is, passing an argument with an incorrect type to an
/// `AnyModule` will compile, but throw an exception at runtime:
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::AnyModule module(torch::nn::Linear(3, 4));
///   // Linear takes a tensor as input, but we are passing an integer.
///   // This will compile, but throw a `torch::Error` exception at runtime.
///   module.forward(123);
/// \endrst
///
/// \rst
/// .. attention::
///   One noteworthy limitation of `AnyModule` is that its `forward()` method
///   does not support implicit conversion of argument types. For example, if
///   the stored module's `forward()` method accepts a `float` and you call
///   `any_module.forward(3.4)` (where `3.4` is a `double`), this will throw
///   an exception.
/// \endrst
///
/// The return type of the `AnyModule`'s `forward()` method is controlled via
/// the first template argument to `AnyModule::forward()`. It defaults to
/// `torch::Tensor`. To change it, you can write `any_module.forward<int>()`,
/// for example.
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::AnyModule module(torch::nn::Linear(3, 4));
///   auto output = module.forward(torch::ones({2, 3}));
///
///   struct IntModule {
///     int forward(int x) { return x; }
///   };
///   torch::nn::AnyModule module(IntModule{});
///   int output = module.forward<int>(5);
/// \endrst
///
/// The only other method an `AnyModule` provides access to on the stored
/// module is `clone()`. However, you may acquire a handle on the module via
/// `.ptr()`, which returns a `shared_ptr<nn::Module>`. Further, if you know
/// the concrete type of the stored module, you can get a concrete handle to it
/// using `.get<T>()` where `T` is the concrete module type.
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::AnyModule module(torch::nn::Linear(3, 4));
///   std::shared_ptr<nn::Module> ptr = module.ptr();
///   torch::nn::Linear linear(module.get<torch::nn::Linear>());
/// \endrst
class AnyModule {
 public:
  /// A default-constructed `AnyModule` is in an empty state.
  AnyModule() = default;

  /// Constructs an `AnyModule` from a `shared_ptr` to concrete module object.
  template <typename ModuleType>
  explicit AnyModule(std::shared_ptr<ModuleType> module);

  /// Constructs an `AnyModule` from a concrete module object.
  template <
      typename ModuleType,
      typename = torch::detail::enable_if_module_t<ModuleType>>
  explicit AnyModule(ModuleType&& module);

  /// Constructs an `AnyModule` from a module holder.
  template <typename ModuleType>
  explicit AnyModule(const ModuleHolder<ModuleType>& module_holder);

  /// Move construction and assignment is allowed, and follows the default
  /// behavior of move for `std::unique_ptr`.
  AnyModule(AnyModule&&) = default;
  AnyModule& operator=(AnyModule&&) = default;

  /// Creates a shallow copy of an `AnyModule`.
  AnyModule(const AnyModule& other);
  AnyModule& operator=(const AnyModule& other);

  /// Creates a deep copy of an `AnyModule` if it contains a module, else an
  /// empty `AnyModule` if it is empty.
  AnyModule clone(optional<Device> device = nullopt) const;

  /// Assigns a module to the `AnyModule` (to circumvent the explicit
  /// constructor).
  template <typename ModuleType>
  AnyModule& operator=(std::shared_ptr<ModuleType> module);

  /// Invokes `forward()` on the contained module with the given arguments, and
  /// returns the return value as an `AnyValue`. Use this method when chaining
  /// `AnyModule`s in a loop.
  template <typename... ArgumentTypes>
  AnyValue any_forward(ArgumentTypes&&... arguments);

  /// Invokes `forward()` on the contained module with the given arguments, and
  /// casts the returned `AnyValue` to the supplied `ReturnType` (which defaults
  /// to `torch::Tensor`).
  template <typename ReturnType = torch::Tensor, typename... ArgumentTypes>
  ReturnType forward(ArgumentTypes&&... arguments);

  /// Attempts to cast the underlying module to the given module type. Throws an
  /// exception if the types do not match.
  template <typename T, typename = torch::detail::enable_if_module_t<T>>
  T& get();

  /// Attempts to cast the underlying module to the given module type. Throws an
  /// exception if the types do not match.
  template <typename T, typename = torch::detail::enable_if_module_t<T>>
  const T& get() const;

  /// Returns the contained module in a `nn::ModuleHolder` subclass if possible
  /// (i.e. if `T` has a constructor for the underlying module type).
  template <typename T, typename ContainedType = typename T::ContainedType>
  T get() const;

  /// Returns a `std::shared_ptr` whose dynamic type is that of the underlying
  /// module.
  std::shared_ptr<Module> ptr() const;

  /// Like `ptr()`, but casts the pointer to the given type.
  template <typename T, typename = torch::detail::enable_if_module_t<T>>
  std::shared_ptr<T> ptr() const;

  /// Returns the `type_info` object of the contained value.
  const std::type_info& type_info() const;

  /// Returns true if the `AnyModule` does not contain a module.
  bool is_empty() const noexcept;

 private:
  /// Creates a `unique_ptr<AnyModulePlaceholder>` pointing to a
  /// `AnyModuleHolder` of the correct type. This method is used to deduce the
  /// arguments of the module's `forward()` method.
  template <
      typename ModuleType,
      typename Class,
      typename ReturnType,
      typename... ArgumentTypes>
  std::unique_ptr<AnyModulePlaceholder> make_holder(
      std::shared_ptr<ModuleType>&& module,
      ReturnType (Class::*)(ArgumentTypes...));

  /// Helper method invoked by const and non-const `get()`.
  template <typename ModuleType, typename ReturnType, typename... ArgumentTypes>
  ModuleType& get_(ReturnType (ModuleType::*)(ArgumentTypes...)) const;

  /// Helper method invoked by const and non-const `get()`.
  template <typename ModuleType>
  ModuleType& get_() const;

  /// The type erased module.
  std::unique_ptr<AnyModulePlaceholder> content_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AnyModule ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename ModuleType>
AnyModule::AnyModule(std::shared_ptr<ModuleType> module)
    : content_(make_holder(
          std::move(module),
          &std::remove_reference<ModuleType>::type::forward)) {
  // `AnyModule` can only store an `nn::Module` subclass object that provides
  // a `forward()` method that has a non-templatized return type.
  // (e.g. `AnyModule` cannot store `nn::Sequential`, because `nn::Sequential`'s
  // `forward()` method has a templatized return type.)
  static_assert(
      torch::detail::is_module<ModuleType>::value,
      "Can only store object derived from nn::Module into AnyModule");
  static_assert(
      torch::detail::has_forward<ModuleType>::value,
      "Can only store module with a forward() method that has a non-templatized"
      " argument type and return type into AnyModule (e.g. we cannot store nn::Sequential"
      "into AnyModule, because its forward() method's argument type and return type are templatized."
      " If you need to use nn::Sequentials inside each other you can subclass "
      "nn::Sequential and write a non-templatized forward function for it. You can checkout "
      "https://github.com/pytorch/vision/blob/2f46070f3cb1ea894d82578f3dc5677f82f34958/torchvision/csrc/models/mnasnet.cpp#L59 "
      "for an example on how to do this.).");
}

template <typename ModuleType, typename>
AnyModule::AnyModule(ModuleType&& module)
    : AnyModule(
          std::make_shared<ModuleType>(std::forward<ModuleType>(module))) {}

template <typename ModuleType>
AnyModule::AnyModule(const ModuleHolder<ModuleType>& module_holder)
    : AnyModule(module_holder.ptr()) {}

inline AnyModule::AnyModule(const AnyModule& other)
    : content_(other.content_ ? other.content_->copy() : nullptr) {}

inline AnyModule& AnyModule::operator=(const AnyModule& other) {
  if (this != &other) {
    content_ = other.content_ ? other.content_->copy() : nullptr;
  }
  return *this;
}

inline AnyModule AnyModule::clone(optional<Device> device) const {
  AnyModule clone;
  clone.content_ = content_ ? content_->clone_module(device) : nullptr;
  return clone;
}

template <typename ModuleType>
AnyModule& AnyModule::operator=(std::shared_ptr<ModuleType> module) {
  // NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature)
  return (*this = AnyModule(std::move(module)));
}

template <typename... ArgumentTypes>
AnyValue AnyModule::any_forward(ArgumentTypes&&... arguments) {
  TORCH_CHECK(!is_empty(), "Cannot call forward() on an empty AnyModule");
  std::vector<AnyValue> values;
  values.reserve(sizeof...(ArgumentTypes));
  torch::apply(
      [&values](AnyValue&& value) { values.push_back(std::move(value)); },
      AnyValue(std::forward<ArgumentTypes>(arguments))...);
  return content_->forward(std::move(values));
}

template <typename ReturnType, typename... ArgumentTypes>
ReturnType AnyModule::forward(ArgumentTypes&&... arguments) {
  return any_forward(std::forward<ArgumentTypes>(arguments)...)
      .template get<ReturnType>();
}

template <typename T, typename>
T& AnyModule::get() {
  TORCH_CHECK(!is_empty(), "Cannot call get() on an empty AnyModule");
  return get_<T>();
}

template <typename T, typename>
const T& AnyModule::get() const {
  TORCH_CHECK(!is_empty(), "Cannot call get() on an empty AnyModule");
  return get_<T>();
}

template <typename T, typename ContainedType>
T AnyModule::get() const {
  return T(ptr<ContainedType>());
}

inline std::shared_ptr<Module> AnyModule::ptr() const {
  TORCH_CHECK(!is_empty(), "Cannot call ptr() on an empty AnyModule");
  return content_->ptr();
}

template <typename T, typename>
std::shared_ptr<T> AnyModule::ptr() const {
  TORCH_CHECK(!is_empty(), "Cannot call ptr() on an empty AnyModule");
  // Call get() but discard the value, just to do the type checking.
  get_<T>();
  return std::dynamic_pointer_cast<T>(ptr());
}

inline const std::type_info& AnyModule::type_info() const {
  TORCH_CHECK(!is_empty(), "Cannot call type_info() on an empty AnyModule");
  return content_->type_info;
}

inline bool AnyModule::is_empty() const noexcept {
  return content_ == nullptr;
}

// Private Methods

template <
    typename ModuleType,
    typename Class,
    typename ReturnType,
    typename... ArgumentTypes>
std::unique_ptr<AnyModulePlaceholder> AnyModule::make_holder(
    std::shared_ptr<ModuleType>&& module,
    ReturnType (Class::*)(ArgumentTypes...)) {
  static_assert(
      torch::detail::check_not_lvalue_references<ArgumentTypes...>(),
      "Modules stored inside AnyModule must not take references. "
      "Use pointers instead.");
  static_assert(
      !std::is_void<ReturnType>::value,
      "AnyModule cannot store modules that return void "
      "(you can return a dummy value).");
  return torch::make_unique<
      AnyModuleHolder<decay_t<ModuleType>, ArgumentTypes...>>(
      std::move(module));
}

template <typename ModuleType>
ModuleType& AnyModule::get_() const {
  using M = typename std::remove_reference<ModuleType>::type;
  static_assert(
      torch::detail::has_forward<M>::value,
      "Can only call AnyModule::get<T> with a type T that has a forward method");
  return get_(&M::forward);
}

template <typename ModuleType, typename ReturnType, typename... ArgumentTypes>
ModuleType& AnyModule::get_(
    ReturnType (ModuleType::*)(ArgumentTypes...)) const {
  if (typeid(ModuleType).hash_code() == type_info().hash_code()) {
    return *static_cast<AnyModuleHolder<ModuleType, ArgumentTypes...>&>(
                *content_)
                .module;
  }
  AT_ERROR(
      "Attempted to cast module of type ",
      c10::demangle(type_info().name()),
      " to type ",
      c10::demangle(typeid(ModuleType).name()));
}

} // namespace nn
} // namespace torch
