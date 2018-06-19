#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <functional>
#include <vector>

namespace torch {
namespace nn {

// Lets you create a container from a function, designed for use in
// Sequential.
class FunctionalImpl : public torch::nn::Cloneable<FunctionalImpl> {
 public:
  using Function = std::function<std::vector<Variable>(std::vector<Variable>)>;

  explicit FunctionalImpl(Function function);
  explicit FunctionalImpl(std::function<Variable(Variable)> function);

  void reset() override;
  std::vector<Variable> forward(std::vector<Variable> input);

 private:
  Function function_;
};

TORCH_MODULE(Functional);

} // namespace nn
} // namespace torch
