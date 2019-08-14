#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd {

struct TORCH_API AccumulateGrad : public Node {
  explicit AccumulateGrad(const Variable& variable_);

  variable_list apply(variable_list&& grads) override;

  WeakVariable variable;
  WeakVariable variable_grad;
};

}} // namespace torch::autograd
