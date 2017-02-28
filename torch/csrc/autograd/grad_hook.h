#pragma once

#include <memory>

// A hook that's called on gradients

namespace torch { namespace autograd {

struct Variable;

struct GradHook {
  virtual std::shared_ptr<Variable> operator()(const std::shared_ptr<Variable>& grad) = 0;
};

}} // namespace torch::autograd
