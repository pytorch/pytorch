#pragma once

#include <torch/csrc/autograd/variable.h>

#include "torch/detail.h"

#include <ATen/optional.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace torch { namespace nn {

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
  virtual std::unique_ptr<Module> clone() const;

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

  std::unordered_map<std::string, std::shared_ptr<nn::Module>> children_;
  std::unordered_map<std::string, Variable> parameters_;

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
  std::shared_ptr<nn::Module> add(
      std::shared_ptr<nn::Module>,
      std::string const&);
  // Be careful when registering Tensors that are not variables
  Variable& add(Variable, std::string const&);

 private:
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

  // should it also detach the gradients, like a deepcopy? Or maybe let's just
  // give clone() a boolean for this?
  std::unique_ptr<Module> clone() const override {
    auto ptr = std::unique_ptr<Module>(
        new Derived(*static_cast<const Derived*>(this)));
    for (auto& parameter : ptr->parameters_) {
      parameter.second = this->parameters_.at(parameter.first).clone();
    }
    for (auto& child : ptr->children_) {
      child.second = this->children_.at(child.first)->clone();
    }
    return ptr;
  }
};

template <class Module>
std::shared_ptr<typename std::decay<Module>::type> make(Module&& module) {
  auto ptr = std::make_shared<typename std::decay<Module>::type>(
      std::forward<Module>(module));
  return ptr;
}
}} // namespace torch::nn
