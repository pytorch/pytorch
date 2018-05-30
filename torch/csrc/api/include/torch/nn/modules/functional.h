#pragma once

#include <torch/nn/module.h>

#include <functional>

namespace torch { namespace nn {

// Lets you create a container from a function, designed for use in
// Sequential.
class Functional : public torch::nn::CloneableModule<Functional> {
 public:
  using Function = std::function<std::vector<Variable>(std::vector<Variable>)>;

  explicit Functional(Function function);
  explicit Functional(std::function<Variable(Variable)> function);

  void reset() override;

  std::vector<Variable> forward(std::vector<Variable> input);

  TORCH_ATTR(Function, function);
};

}} // namespace torch::nn
