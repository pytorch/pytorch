#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd {

struct TORCH_API AccumulateGrad : public Node {
  explicit AccumulateGrad(Variable variable_);

  variable_list apply(variable_list&& grads) override;

  // Given a variable with its current grad as variable_grad, accumulates
  // new_grad into variable_grad.
  static void accumulateGradAndCallHooks(
      const Variable& variable,
      at::Tensor variable_grad,
      at::Tensor new_grad,
      bool has_post_hooks,
      std::function<void(at::Tensor)> update_grad_fn);

  Variable variable;
};

}} // namespace torch::autograd
