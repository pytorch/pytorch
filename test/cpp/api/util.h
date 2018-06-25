#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/tensor.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace torch {

// Lets you use a container without making a new class,
// for experimental implementations
class SimpleContainer : public nn::Cloneable<SimpleContainer> {
 public:
  void reset() override {}

  template <typename ModuleHolder>
  ModuleHolder add(
      ModuleHolder module_holder,
      std::string name = std::string()) {
    return Module::register_module(std::move(name), module_holder);
  }
};

struct SigmoidLinear : nn::Module {
  SigmoidLinear(int64_t in, int64_t out) : linear(nn::Linear(in, out)) {
    register_module("linear", linear);
  }

  explicit SigmoidLinear(nn::Linear linear_) : linear(std::move(linear_)) {
    register_module("linear", linear);
  }
  Tensor forward(Tensor input) {
    return linear->forward(input).sigmoid();
  }
  nn::Linear linear;
};

} // namespace torch
