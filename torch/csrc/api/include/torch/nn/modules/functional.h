#pragma once

#include <torch/nn/module.h>

#include <functional>

namespace torch { namespace nn {

class Functional : public torch::nn::CloneableModule<Functional> {
  // Lets you create a container from a function, designed for use in
  // Sequential.
 public:
  Functional(std::function<variable_list(variable_list)> fun) : fun_(fun){};
  Functional(std::function<Variable(Variable)> fun)
      : fun_([fun](variable_list input) {
          return variable_list({fun(input[0])});
        }){};

  variable_list forward(variable_list input) override {
    return fun_(input);
  };

  std::function<variable_list(variable_list)> fun_;
};

}} // namespace torch::nn
