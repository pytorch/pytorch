#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace torch {

// Lets you use a container without making a new class,
// for experimental implementations
class SimpleContainer : public nn::CloneableModule<SimpleContainer> {
 public:
  virtual std::vector<Variable> forward(std::vector<Variable>) {
    throw std::runtime_error(
        "SimpleContainer has no forward, maybe you"
        " wanted to subclass and override this function?");
  }

  void reset() {}

  template <typename Derived>
  std::shared_ptr<Derived> add(
      std::shared_ptr<Derived> module,
      std::string name = std::string()) {
    return Module::register_module(std::move(name), module);
  }
};

struct SigmoidLinear : nn::Module {
  SigmoidLinear(size_t in, size_t out) : linear(nn::Linear(in, out).build()) {
    register_module("linear", linear);
  }

  explicit SigmoidLinear(std::shared_ptr<nn::Linear> linear_)
      : linear(std::move(linear_)) {
    register_module("linear", linear);
  }
  Variable forward(Variable input) {
    return linear->forward({input}).front().sigmoid();
  }
  std::shared_ptr<nn::Linear> linear;
};

} // namespace torch
