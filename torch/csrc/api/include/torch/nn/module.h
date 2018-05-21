#pragma once

#include "torch/detail.h"

#include <torch/detail/member_ref.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/optional.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

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

  virtual variable_list forward(variable_list) = 0;
  virtual std::shared_ptr<Module> clone() const;

  std::map<std::string, Variable> parameters() const;
  Variable& param(std::string const&);

  /// Enables training mode.
  virtual void train();

  /// Disables training mode.
  virtual void eval();

  /// True if the module is in training mode.
  virtual bool is_training() const noexcept;

  /// Recursively moves all parameters to CPU memory (in place).
  virtual void cpu();

  /// Recursively moves all parameters to CUDA memory (in place).
  virtual void cuda();

  /// Recursively casts all parameters to the given type.
  virtual void to(at::Type& type);

  /// Recursively casts all parameters to the given scalar type.
  virtual void to(at::ScalarType scalar_type);

  /// Recursively moves all parameters to the given backend.
  virtual void to(at::Backend backend);

  /// Recursively zeros out the `grad` values of all parameters.
  virtual void zero_grad();

  template <class Archive>
  void save(Archive& ar) const {
    auto params = parameters();
    std::size_t size = params.size();
    ar(size);
    for (auto& p : params) {
      ar(p.first, p.second);
    }
  }

  template <class Archive>
  void load(Archive& ar) {
    auto params = parameters();
    std::size_t size;
    ar(size);
    std::string name;
    for (std::size_t i = 0; i < size; i++) {
      ar(name);
      ar(params[name]);
    }
  }

 protected:
  using ModulePtr = std::shared_ptr<nn::Module>;

  template <typename Derived, typename M>
  void register_module(
      const std::string& name,
      std::shared_ptr<M> Derived::*module,
      std::shared_ptr<M> new_module) {
    check_name("Module", name);
    auto base_module =
        reinterpret_cast<std::shared_ptr<nn::Module> Derived::*>(module);
    const auto pair = children_.emplace(name, base_module);
    AT_CHECK(pair.second, "Module " + name + " has already been registered");
    (pair.first->second)(static_cast<Derived*>(this)) = std::move(new_module);
  }

  // Hack for Sequential, should remove/figure out to do this better
  template <typename Derived>
  void register_module(
      const std::string& name,
      std::vector<std::shared_ptr<Module>> Derived::*modules,
      size_t index) {
    const auto pair = children_.insert({name, {modules, index}});
    AT_CHECK(pair.second, "Module " + name + " has already been registered");
  }

  template <typename Derived>
  void register_parameter(
      const std::string& name,
      Variable Derived::*variable,
      Tensor tensor) {
    check_name("Parameter", name);
    const auto pair = parameters_.emplace(name, variable);
    AT_CHECK(pair.second, "Parameter " + name + " has already been registered");
    (pair.first->second)(static_cast<Derived*>(this)) =
        autograd::make_variable(tensor, /*requires_grad=*/true);
  }

  template <typename Derived>
  void register_buffer(
      const std::string& name,
      Variable Derived::*variable,
      Tensor tensor) {
    check_name("Parameter", name);
    const auto pair = parameters_.emplace(name, variable);
    AT_CHECK(pair.second, "Parameter " + name + " has already been registered");
    (pair.first->second)(static_cast<Derived*>(this)) =
        autograd::make_variable(tensor, /*requires_grad=*/false);
  }

  template <typename Getter, typename ConstGetter>
  void register_parameter(
      const std::string& name,
      Getter getter,
      ConstGetter const_getter) {
    check_name("Parameter", name);
    const auto pair = parameters_.insert(
        {name, {std::move(getter), std::move(const_getter)}});
    AT_CHECK(pair.second, "Parameter " + name + " has already been registered");
  }

  void check_name(const std::string& type, const std::string& name) {
    AT_CHECK(!name.empty(), type + " name must not be empty");
    AT_CHECK(
        name.find('.') == std::string::npos,
        type + " name '" + name + "' contains a dot, which is not permitted");
  }

 private:
  template <typename Derived>
  friend class CloneableModule;

  std::unordered_map<std::string, detail::MemberRef<Variable>> parameters_;
  std::unordered_map<std::string, detail::MemberRef<ModulePtr>> children_;

  /// The module's name (e.g. "LSTM").
  mutable at::optional<std::string> name_;

  /// Whether the module is in training mode.
  bool is_training_{true};
};

/// The `clone()` method in the base `Module` class does not have knowledge of
/// the concrete runtime type of its subclasses. Therefore, `clone()` must
/// either be called from within the subclass, or from a base class that has
/// knowledge of the concrete type. `CloneableModule` uses the CRTP to gain
/// knowledge of the subclass' static type and provide an implementation of the
/// `clone()` method. We do not want to use this pattern in the base class,
/// because then storing a module would always require templatizing it.
template <typename Derived>
class CloneableModule : public Module {
 public:
  using Module::Module;

  virtual void reset() = 0;

  /// Moves the `Module` into a `shared_ptr` and calls `reset()` on it.
  std::shared_ptr<Derived> build() {
    auto module = std::make_shared<Derived>(static_cast<Derived&&>(*this));
    module->reset();
    return std::move(module);
  }

  /// Performs a recursive "deep copy" of the `Module`, such that all parameters
  /// and submodules in the cloned module are different from those in the
  /// original module.
  std::shared_ptr<Module> clone() const override {
    auto ptr = std::make_shared<Derived>(*static_cast<const Derived*>(this));
    for (auto& pair : ptr->parameters_) {
      auto& parameter = pair.second(ptr.get());
      auto& original = this->parameters_.at(pair.first)(this);
      parameter = at::empty_like(original);
      parameter.data().copy_(original.data());
    }
    for (auto& child : ptr->children_) {
      child.second(ptr.get()) = this->children_.at(child.first)(this)->clone();
    }
    return ptr;
  }
};
} // namespace nn
} // namespace torch

#define TORCH_ATTR(T, name)                         \
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
