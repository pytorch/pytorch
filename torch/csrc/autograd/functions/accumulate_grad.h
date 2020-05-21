#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <mutex>

namespace torch { namespace autograd {

struct TORCH_API AccumulateGrad : public Node {
  explicit AccumulateGrad(Variable variable_);

  variable_list apply(variable_list&& grads) override;

  static at::Tensor callHooks(
      const Variable& variable,
      at::Tensor new_grad) {
    for (auto& hook : impl::hooks(variable)) {
      new_grad = (*hook)({new_grad})[0];
    }
    return new_grad;
  }

  // Given a variable with its current grad as variable_grad, accumulates
  // new_grad into variable_grad if in place accumulation is possible.
  // Otherwise, uses 'update_grad' to update the grad for the variable.

  // "Gradient Layout Contract"
  //
  // AccumulateGrad tries to stash grads with memory layout (strides) such
  // that variables and grads interact efficiently in later optimizer kernels,
  // and grads interact efficiently with c10d::Reducer.cpp.
  //
  // Specifically, for strided (non-sparse) gradients, AccumulateGrad tries to
  // ensure the following:
  //   (1) if variable.is_non_overlapping_and_dense(), the stashed grad's
  //       sizes and strides match variable.
  //   (2) if variable's memory isn't dense, the stashed grad's sizes match
  //       variable, but it falls back to the rowmajor-contiguous strides.
  //
  // If variable's grad already exists (variable_grad.defined() is true),
  // AccumulateGrad assumes (hopes) variable_grad was created in the past
  // according to (1) or (2), and stashes variable_grad + new_grad
  // Therefore, AccumulateGrad does not enforce (1) and (2) with 100%
  // certainty.  For example, if variable_grad was assigned by the user,
  // such that AccumulateGrad never got the chance to create it, variable_grad
  // enters operator+ with whatever layout the user gave it, and we are
  // at the mercy of TensorIterator to decide what the layout of
  // "variable_grad + new_grad" will be.
  //
  // Fortunately, if a given grad doesn't satisfy (1) or (2), the penalty is
  // degraded performance in Reducer.cpp or optimizer kernels, not death by
  // assert or silently bad numerics.
  //
  // We could add logic that is more aggressive about forcing
  // "variable_grad + new_grad" to obey (1) or (2).

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
  static void accumulateGrad(
      const Variable& variable,
      at::Tensor& variable_grad,
      const at::Tensor& new_grad,
      size_t num_expected_refs,
      const T& update_grad) {
    if (!variable_grad.defined()) {
      // under following conditions, we can avoid deep copy (steal the gradient)
      if (!GradMode::is_enabled() &&
          !new_grad.is_sparse() &&
          new_grad.use_count() <= num_expected_refs &&
          (variable.is_non_overlapping_and_dense() ?
           (new_grad.strides() == variable.strides()) :
           new_grad.is_contiguous())) {
        // first check we aren't setting up for double-backward
        // then check not sparse
        // then check if use_count <= number of references we expect
        // then check if new_grad obeys the "Gradient Layout Contract"
        update_grad(new_grad.detach());
      } else if (
          !GradMode::is_enabled() && new_grad.is_sparse() &&
          new_grad._indices().is_contiguous() &&
          new_grad._values().is_contiguous() &&
          // Use count for indices and values should always be <=1 since the
          // SparseTensor should be the only one holding a reference to these.
          new_grad._indices().use_count() <= 1 &&
          new_grad._values().use_count() <= 1 &&
          new_grad.use_count() <= num_expected_refs) {
        // Can't detach sparse tensor (since metadata changes are not allowed
        // after detach), so just create a new one for the grad which is a
        // shallow copy. We need a shallow copy so that modifying the original
        // grad tensor doesn't modify the grad we accumulate.
        // We only skip clone if indices and values themselves are contiguous
        // for backward compatiblity reasons. Since without this optimization,
        // earlier we would clone the entire SparseTensor which cloned indices
        // and values.
        // For details see https://github.com/pytorch/pytorch/issues/34375.
        update_grad(at::_sparse_coo_tensor_unsafe(
            new_grad._indices(),
            new_grad._values(),
            new_grad.sizes(),
            new_grad.options()));
      } else {
        if (new_grad.is_sparse()) {
          update_grad(new_grad.clone());
        } else {
          // Deep copies new_grad according to the "Gradient Layout Contract."
          if (variable.is_non_overlapping_and_dense()) {
            if (variable.strides() == new_grad.strides()) {
              update_grad(new_grad.clone());
            } else {
              update_grad(std::move(at::empty_strided(variable.sizes(), variable.strides(),
                                    variable.options().memory_format(c10::nullopt))
                                    .copy_(new_grad)));
            }
          } else {
            update_grad(new_grad.clone(at::MemoryFormat::Contiguous));
          }
        }
      }
    } else if (!GradMode::is_enabled()) {
      // This case is not strictly necessary, but it makes the first-order only
      // case slightly more efficient.
      if (variable_grad.is_sparse() && !new_grad.is_sparse()) {
        // If `variable_grad` is sparse and `new_grad` is not sparse, their
        // sum is not sparse, and we must change the TensorImpl type of
        // `variable_grad` for it to store the result. However, changing the
        // TensorImpl type of a tensor requires changing the tensor itself, and
        // thus in this case we have to change the grad tensor.
        update_grad(new_grad + variable_grad);
      } else {
        // In this case we can avoid changing the grad tensor. There are three
        // scenarios when we'll hit this case:
        //
        // 1. `variable_grad` is sparse, and `new_grad` is sparse.
        // 2. `variable_grad` is dense, and `new_grad` is sparse.
        // 3. `variable_grad` is dense, and `new_grad` is dense.
        //
        // In all of these three cases, `variable_grad += new_grad` is a
        // valid operation which adds `new_grad` to `variable_grad` in
        // place. `variable_grad` is thus still referring to the same tensor
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
