#pragma once

#include <torch/nn/modules/container/any_value.h>

namespace torch {
namespace nn {

class Module;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~ AnyModulePlaceholder ~~~~~~~~~~~~~~~~~~~~~~~~~~

/// The static type of the object we store in the `AnyModule`, which erases
/// the actual type, but allows us to call `forward()` on the underlying
/// module.
struct AnyModulePlaceholder : public AnyValue::Placeholder {
  using AnyValue::Placeholder::Placeholder;

  /// The "erased" `forward()` method.
  virtual AnyValue forward(std::vector<AnyValue>&& arguments) = 0;

  /// Returns std::shared_ptr<Module> pointing to the erased module.
  virtual std::shared_ptr<Module> ptr() = 0;

  /// Returns a `AnyModulePlaceholder` with a shallow copy of this `AnyModule`.
  virtual std::unique_ptr<AnyModulePlaceholder> copy() const = 0;

  /// Returns a `AnyModulePlaceholder` with a deep copy of this `AnyModule`.
  virtual std::unique_ptr<AnyModulePlaceholder> clone_module(
      optional<Device> device) const = 0;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AnyModuleHolder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// The dynamic type of the object stored in the `AnyModule`. It contains the
/// concrete instance to which all calls are forwarded. It is parameterized
/// over the concrete type of the module, and the types of the arguments the
/// module takes in its `forward()` method.
template <typename ModuleType, typename... ArgumentTypes>
struct AnyModuleHolder : public AnyModulePlaceholder {
  /// \internal
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
          c10::demangle(typeid(T).name()),
          ", but received value of type ",
          c10::demangle(value.type_info().name()));
    }
    std::vector<AnyValue>& arguments_;
  };

  /// \internal
  struct InvokeForward {
    template <typename... Ts>
    AnyValue operator()(Ts&&... ts) {
      return AnyValue(module_->forward(std::forward<Ts>(ts)...));
    }
    std::shared_ptr<ModuleType>& module_;
  };

  /// Constructs the `AnyModuleHolder` from a concrete module.
  explicit AnyModuleHolder(std::shared_ptr<ModuleType>&& module_)
      : AnyModulePlaceholder(typeid(ModuleType)), module(std::move(module_)) {}

  /// Calls `forward()` on the underlying module, casting each `AnyValue` in the
  /// argument vector to a concrete value.
  AnyValue forward(std::vector<AnyValue>&& arguments) override {
    if (module->_forward_has_default_args()) {
      TORCH_CHECK(
          arguments.size() >= module->_forward_num_required_args() &&
              arguments.size() <= sizeof...(ArgumentTypes),
          c10::demangle(type_info.name()),
          "'s forward() method expects at least ",
          module->_forward_num_required_args(),
          " argument(s) and at most ",
          sizeof...(ArgumentTypes),
          " argument(s), but received ",
          arguments.size(),
          ".");
      arguments = std::move(
          module->_forward_populate_default_args(std::move(arguments)));
    } else {
      std::string use_default_args_macro_prompt = " If " +
          c10::demangle(type_info.name()) +
          "'s forward() method has default arguments, " +
          "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.";
      TORCH_CHECK(
          arguments.size() == sizeof...(ArgumentTypes),
          c10::demangle(type_info.name()),
          "'s forward() method expects ",
          sizeof...(ArgumentTypes),
          " argument(s), but received ",
          arguments.size(),
          ".",
          (arguments.size() < sizeof...(ArgumentTypes))
              ? use_default_args_macro_prompt
              : "");
    }

    // FYI: During invocation of a module's `forward()` method, the values live
    // in the `arguments` vector inside this function.
    return torch::unpack<AnyValue, ArgumentTypes...>(
        InvokeForward{module}, CheckedGetter{arguments});
  }

  std::shared_ptr<Module> ptr() override {
    return module;
  }

  std::unique_ptr<AnyModulePlaceholder> copy() const override {
    return torch::make_unique<AnyModuleHolder>(*this);
  }

  std::unique_ptr<AnyModulePlaceholder> clone_module(
      optional<Device> device) const override {
    return torch::make_unique<AnyModuleHolder>(
        std::dynamic_pointer_cast<ModuleType>(module->clone(device)));
  }

  /// The actual concrete module instance.
  std::shared_ptr<ModuleType> module;
};

} // namespace nn
} // namespace torch
