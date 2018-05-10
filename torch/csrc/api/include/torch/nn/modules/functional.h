#pragma once

#include <torch/nn/module.h>

#include <functional>

namespace torch { namespace nn {

// Lets you create a container from a function, designed for use in
// Sequential.
class Functional : public torch::nn::CloneableModule<Functional> {
 public:
  explicit Functional(std::function<variable_list(variable_list)> fun);
  explicit Functional(std::function<Variable(Variable)> fun);

  variable_list forward(variable_list input) override;

  std::function<variable_list(variable_list)> fun_;
};

}} // namespace torch::nn
