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
  // variable: the variable whose grad we're accumulating.
  // variable_grad: the current grad for the variable.
  // new_grad: new grad we want to acummulate for the variable.
  // num_expected_refs: the number of refs we expect to hold internally
  //                    such that it is safe to avoid cloning the grad
  //                    if use_count() of the grad is less than or equal
  //                    to this value (in addition to post_hooks).
  // has_post_hooks: Whether or post_hooks are holding references to
  //                 the grad tensor, this is also used to determine
  //                 whether or not to clone.
  static void accumulateGradAndCallHooks(
      const Variable& variable,
      at::Tensor variable_grad,
      const at::Tensor& new_grad,
      size_t num_expected_refs,
      const std::function<void(at::Tensor&&)>& update_grad_fn);

  Variable variable;
};

} // namespace autograd
} // namespace torch
