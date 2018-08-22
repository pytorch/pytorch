#pragma once

#include <torch/detail/static.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/memory.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/Device.h>
#include <ATen/core/optional.h>

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

/// A class to store a type erased module, whose `forward()` method can be
/// invoked, with dynamic type checking. An `AnyModule` has an empty state, into
/// which it is default constructed. `is_empty()` can be used to query whether
/// the `AnyModule` is empty.
class AnyModule {
 public:
  /// A type-erased value.
  class Value;

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
  AnyModule clone(at::optional<Device> device = at::nullopt) const;

  /// Assigns a module to the `AnyModule` (to circumvent the explicit
  /// constructor).
  template <typename ModuleType>
  AnyModule& operator=(std::shared_ptr<ModuleType> module);

  /// Invokes `forward()` on the contained module with the given arguments, and
  /// returns the return value as an `Value`. Use this method when chaining
  /// `AnyModule`s in a loop.
  template <typename... ArgumentTypes>
  Value forward(ArgumentTypes&&... arguments);

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
  /// The static type of the object we store in the `AnyModule`, which erases
  /// the actual type, but allows us to call `forward()` on the underlying
  /// module.
  struct Placeholder;

  /// The dynamic type of the object stored in the `AnyModule`. It contains the
  /// concrete instance to which all calls are forwarded. It is parameterized
  /// over the concrete type of the module, and the types of the arguments the
  /// module takes in its `forward()` method.
  template <typename ModuleType, typename... ArgumentTypes>
  struct Holder;

  /// Creates a `unique_ptr<Placeholder>` pointing to a `Holder` of the correct
  /// type. This method is used to deduce the arguments of the module's
  /// `forward()` method.
  template <
      typename ModuleType,
      typename Class, // = std::remove_reference<ModuleType>::type
      typename ReturnType,
      typename... ArgumentTypes>
  std::unique_ptr<Placeholder> make_holder(
      std::shared_ptr<ModuleType>&& module,
      ReturnType (Class::*)(ArgumentTypes...));

  /// Helper method invoked by const and non-const `get()`.
  template <typename T>
  T& get_() const;

  /// The type erased module.
  std::unique_ptr<Placeholder> content_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AnyModule::Value ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A simplified implementation of `std::any` which stores
/// a type erased object, whose concrete value can be retrieved at runtime by
/// checking if the `typeid()` of a requested type matches the `typeid()` of
/// the object stored. It is simplified in that it does not handle copying, as
/// we do not require it for our use cases. Moves are sufficient.
class AnyModule::Value {
 public:
  /// Move construction and assignment is allowed, and follows the default
  /// behavior of move for `std::unique_ptr`.
  Value(Value&&) = default;
  Value& operator=(Value&&) = default;

  /// Copy is disallowed, because we don't need it.
  Value(const Value& other) = delete;
  Value& operator=(const Value& other) = delete;

  /// Returns a pointer to the value contained in the `Value` if the type passed
  /// as template parameter matches the type of the value stored, and returns a
  /// null pointer otherwise.
  template <typename T>
  T* try_get() {
    static_assert(
        !std::is_reference<T>::value,
        "Value stores decayed types, you cannot cast it to a reference type");
    static_assert(
        !std::is_array<T>::value,
        "Value stores decayed types, you must cast it to T* instead of T[]");
    if (typeid(T).hash_code() == type_info().hash_code()) {
      return &static_cast<Holder<T>&>(*content_).value;
    }
    return nullptr;
  }

  /// Returns the value contained in the `Value` if the type passed as template
  /// parameter matches the type of the value stored, and throws an exception
  /// otherwise.
  template <typename T>
  T get() {
    if (auto* maybe_value = try_get<T>()) {
      return *maybe_value;
    }
    AT_ERROR(
        "Attempted to cast Value to ",
        at::demangle(typeid(T).name()),
        ", but its actual type is ",
        at::demangle(type_info().name()));
  }

  /// Returns the `type_info` object of the contained value.
  const std::type_info& type_info() const noexcept {
    return content_->type_info;
  }

 private:
  friend class AnyModule;
  friend struct TestValue;

  /// Constructs the `Value` from value type.
  template <
      typename T,
      typename =
          torch::disable_if_t<std::is_same<autograd::Variable, T>::value>>
  explicit Value(T&& value)
      : content_(
            torch::make_unique<Holder<decay_t<T>>>(std::forward<T>(value))) {}

  /// Constructs the `Value` from an `autograd::Variable`, first converting it
  /// to a `torch::Tensor`.
  explicit Value(autograd::Variable variable)
      : Value(Tensor(std::move(variable))) {}

  /// The static type of the object we store in the `Value`, which erases the
  /// actual object's type, allowing us only to check the `type_info` of the
  /// type stored in the dynamic type.
  struct Placeholder {
    explicit Placeholder(const std::type_info& type_info_) noexcept
        : type_info(type_info_) {}
    virtual ~Placeholder() = default;
    const std::type_info& type_info;
  };

  /// The dynamic type of the object we store in the `Value`, which hides the
  /// actual object we have erased in this `Value`.
  template <typename T>
  struct Holder : public Placeholder {
    /// A template because T&& would not be universal reference here.
    template <typename U>
    explicit Holder(U&& value_) noexcept
        : Placeholder(typeid(T)), value(std::forward<U>(value_)) {}
    T value;
  };

  /// The type erased object.
  std::unique_ptr<Placeholder> content_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~ AnyModule::Placeholder ~~~~~~~~~~~~~~~~~~~~~~~~~~

struct AnyModule::Placeholder : public AnyModule::Value::Placeholder {
  using AnyModule::Value::Placeholder::Placeholder;

  /// The "erased" `forward()` method.
  virtual Value forward(std::vector<Value>&& arguments) = 0;

  /// Returns std::shared_ptr<Module> pointing to the erased module.
  virtual std::shared_ptr<Module> ptr() = 0;

  /// Returns a `Placeholder` with a shallow copy of this `AnyModule`.
  virtual std::unique_ptr<Placeholder> copy() const = 0;

  /// Returns a `Placeholder` with a deep copy of this `AnyModule`.
  virtual std::unique_ptr<Placeholder> clone(
      at::optional<Device> device) const = 0;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AnyModule::Holder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename ModuleType, typename... ArgumentTypes>
struct AnyModule::Holder : public AnyModule::Placeholder {
  struct CheckedGetter {
    template <typename T>
    decay_t<T>&& operator()(size_t index) {
      AT_ASSERT(index < arguments_.size());
      auto& value = arguments_[index];
      if (auto* maybe_value = value.template try_get<decay_t<T>>()) {
        return std::move(*maybe_value);
      }
      AT_ERROR(
          "Expected argument #",
          index,
          " to be of type ",
          at::demangle(typeid(T).name()),
          ", but received value of type ",
          at::demangle(value.type_info().name()));
    }
    std::vector<Value>& arguments_;
  };

  struct InvokeForward {
    template <typename... Ts>
    Value operator()(Ts&&... ts) {
      return Value(module_->forward(std::forward<Ts>(ts)...));
    }
    std::shared_ptr<ModuleType>& module_;
  };

  /// Constructs the `Holder` from a concrete module.
  explicit Holder(std::shared_ptr<ModuleType>&& module_)
      : Placeholder(typeid(ModuleType)), module(std::move(module_)) {}

  /// Calls `forward()` on the underlying module, casting each `Value` in the
  /// argument vector to a concrete value.
  Value forward(std::vector<Value>&& arguments) override {
    AT_CHECK(
        arguments.size() == sizeof...(ArgumentTypes),
        at::demangle(type_info.name()),
        "'s forward() method expects ",
        sizeof...(ArgumentTypes),
        " arguments, but received ",
        arguments.size());
    // FYI: During invocation of a module's `forward()` method, the values live
    // in the `arguments` vector inside this function.
    return torch::unpack<ArgumentTypes...>(
        InvokeForward{module}, CheckedGetter{arguments});
  }

  std::shared_ptr<Module> ptr() override {
    return module;
  }

  std::unique_ptr<Placeholder> copy() const override {
    return torch::make_unique<Holder>(*this);
  }

  std::unique_ptr<Placeholder> clone(
      at::optional<Device> device) const override {
    return torch::make_unique<Holder>(
        std::dynamic_pointer_cast<ModuleType>(module->clone(device)));
  }

  /// The actual concrete module instance.
  std::shared_ptr<ModuleType> module;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AnyModule ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename ModuleType>
AnyModule::AnyModule(std::shared_ptr<ModuleType> module)
    : content_(make_holder(
          std::move(module),
          &std::remove_reference<ModuleType>::type::forward)) {}

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

inline AnyModule AnyModule::clone(at::optional<Device> device) const {
  AnyModule clone;
  clone.content_ = content_ ? content_->clone(device) : nullptr;
  return clone;
}

template <typename ModuleType>
AnyModule& AnyModule::operator=(std::shared_ptr<ModuleType> module) {
  return (*this = AnyModule(std::move(module)));
}

template <typename... ArgumentTypes>
AnyModule::Value AnyModule::forward(ArgumentTypes&&... arguments) {
  AT_CHECK(!is_empty(), "Cannot call forward() on an empty AnyModule");
  std::vector<Value> values;
  values.reserve(sizeof...(ArgumentTypes));
  torch::apply(
      [&values](Value&& value) { values.push_back(std::move(value)); },
      Value(std::forward<ArgumentTypes>(arguments))...);
  return content_->forward(std::move(values));
}

template <typename T, typename>
T& AnyModule::get() {
  AT_CHECK(!is_empty(), "Cannot call get() on an empty AnyModule");
  return get_<T>();
}

template <typename T, typename>
const T& AnyModule::get() const {
  AT_CHECK(!is_empty(), "Cannot call get() on an empty AnyModule");
  return get_<T>();
}

template <typename T, typename ContainedType>
T AnyModule::get() const {
  return T(ptr<ContainedType>());
}

inline std::shared_ptr<Module> AnyModule::ptr() const {
  AT_CHECK(!is_empty(), "Cannot call ptr() on an empty AnyModule");
  return content_->ptr();
}

template <typename T, typename>
std::shared_ptr<T> AnyModule::ptr() const {
  AT_CHECK(!is_empty(), "Cannot call ptr() on an empty AnyModule");
  /// Call get() but discard the value, just to do the type checking.
  get_<T>();
  return std::static_pointer_cast<T>(ptr());
}

inline const std::type_info& AnyModule::type_info() const {
  AT_CHECK(!is_empty(), "Cannot call type_info() on an empty AnyModule");
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
std::unique_ptr<AnyModule::Placeholder> AnyModule::make_holder(
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
  return torch::make_unique<Holder<decay_t<ModuleType>, ArgumentTypes...>>(
      std::move(module));
}

template <typename T>
T& AnyModule::get_() const {
  if (typeid(T).hash_code() == type_info().hash_code()) {
    return *static_cast<Holder<T>&>(*content_).module;
  }
  AT_ERROR(
      "Attempted to cast module of type ",
      at::demangle(type_info().name()),
      " to type ",
      at::demangle(typeid(T).name()));
}

} // namespace nn
} // namespace torch
