#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd {

struct TORCH_API AccumulateGrad : public Node {
  explicit AccumulateGrad(Variable variable_);

  variable_list apply(variable_list&& grads) override;

  // Given a variable with its current grad as variable_grad, accumulates
  // new_grad into variable_grad if in place accumulation is possible.
  // Otherwise, uses 'update_grad' to update the grad for the variable.

  // variable: the variable whose grad we're accumulating.
  // variable_grad: the current grad for the variable.
  // new_grad: new grad we want to acummulate for the variable.
  // num_expected_refs: the number of refs we expect to hold internally
  //                    such that it is safe to avoid cloning the grad
  //                    if use_count() of the grad is less than or equal
  //                    to this value (in addition to post_hooks).
  // update_grad: Function that is used to update grad for the variable.
  //              The argument to the function is a Tensor which
  //              is used to set a new value for the grad.
  template <typename T>
  static void accumulateGradAndCallHooks(
      const Variable& variable,
      at::Tensor& variable_grad,
      at::Tensor new_grad,
      size_t num_expected_refs,
      const T& update_grad) {
    for (auto& hook : impl::hooks(variable)) {
      new_grad = (*hook)({new_grad})[0];
    }

    if (!variable_grad.defined()) {
      // under following condition, we can avoid clone()
      if (!GradMode::is_enabled() && !new_grad.is_sparse() &&
          new_grad.is_contiguous() &&
          new_grad.use_count() <= num_expected_refs) {
        // first check it is in first-order grad only mode
        // then check not sparse before is_contiguous
        // then check contiguous, otherwise later in place accumulation may fail
        // and lastly, check if the use_count is less than or equal to the
        // number of references we expect before grabbing it. The number of
        // references we expect is basically internal structures that are
        // holding references to the Tensor and that is fine since these are not
        // exposed to the user.
        update_grad(new_grad.detach());
      } else {
        if (new_grad.is_sparse()) {
          update_grad(new_grad.clone());
        } else {
          update_grad(new_grad.clone(at::MemoryFormat::Contiguous));
        }
      }
    } else if (!GradMode::is_enabled()) {
      // This case is not strictly necessary, but it makes the first-order only
      // case slightly more efficient.
      if (variable_grad.is_sparse() && !new_grad.is_sparse()) {
        // If `grad_variable` is sparse and `new_grad` is not sparse, their
        // sum is not sparse, and we must change the TensorImpl type of
        // `grad_variable` for it to store the result. However, changing the
        // TensorImpl type of a tensor requires changing the tensor itself, and
        // thus in this case we have to change the grad tensor.
        update_grad(new_grad + variable_grad);
      } else {
        // In this case we can avoid changing the grad tensor. There are three
        // scenarios when we'll hit this case:
        //
        // 1. `grad_variable` is sparse, and `new_grad` is sparse.
        // 2. `grad_variable` is dense, and `new_grad` is sparse.
        // 3. `grad_variable` is dense, and `new_grad` is dense.
        //
        // In all of these three cases, `grad_variable += new_grad` is a
        // valid operation which adds `new_grad` to `grad_variable` in
        // place. `grad_variable` is thus still referring to the same tensor
        // after the operation.
        variable_grad += new_grad;
      }
    } else {
      update_grad(variable_grad + new_grad);
    }
  }

  Variable variable;
};

} // namespace autograd
} // namespace torch
