#pragma once

#include <torch/csrc/autograd/variable.h>

#include "torch/detail.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace torch { namespace nn {

class Module {
 public:
  virtual ~Module() = default;

  // Only construct parameters in initialize_parameters, and
  // containers in initialize_containers. Most of the time, the containers are
  // the only thing you need to add.
  // You are guaranteed that containers are added before parameters.
  virtual void initialize_containers(){};
  virtual void initialize_parameters(){};
  virtual void reset_parameters(){};

  virtual variable_list forward(variable_list) = 0;
  virtual std::unique_ptr<Module> clone() const = 0;

  std::map<std::string, Variable> parameters() const;
  Variable& param(std::string const&);

  virtual void cuda();
  virtual void cpu();
  void train();
  void eval();

  at::Type& DefaultTensor(at::ScalarType s);

  std::unordered_map<std::string, std::shared_ptr<nn::Module>> children_;
  std::unordered_map<std::string, Variable> params_;
  bool cuda_ = false;
  bool train_ = true;

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
  // explicit CloneableModule(const char* name) : Module(name) {}

  std::unique_ptr<Module> clone() const override {
    auto ptr = std::unique_ptr<Module>(
        new Derived(*static_cast<const Derived*>(this)));
    ptr->children_.clear();
    ptr->params_.clear();
    ptr->initialize_containers();
    ptr->initialize_parameters();
    auto newParams = ptr->parameters();
    for (auto& param : parameters()) {
      newParams[param.first].data().copy_(param.second.data());
    }
    if (cuda_) {
      ptr->cuda();
    } else {
      ptr->cpu();
    }
    return ptr;
  }

  // return std::unique_ptr<Module>(new
  // Derived(static_cast<Derived&>(*this)));
};

template <class Module>
std::shared_ptr<typename std::decay<Module>::type> make(Module&& module) {
  auto ptr = std::make_shared<typename std::decay<Module>::type>(
      std::forward<Module>(module));
  ptr->initialize_containers();
  ptr->initialize_parameters();
  ptr->reset_parameters();
  return ptr;
}
}} // namespace torch::nn
