#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/utils/grad_layout_contract.h>
#include <torch/csrc/Export.h>
#include <ATen/BatchedTensorImpl.h>
#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#endif

#include <mutex>

namespace torch { namespace autograd {

#define CHECK_RESULT(RESULT, VAR) \
  if (!(RESULT.is_sparse() || VAR.is_sparse())) { \
    if (!utils::obeys_layout_contract(RESULT, VAR)) { \
      TORCH_WARN_ONCE("grad and param do not obey the gradient layout contract. " \
                      "This is not an error, but may impair performance.\n" \
                      "grad.sizes() = ", RESULT.sizes(), \
                      ", strides() = ", RESULT.strides(), "\n", \
                      "param.sizes() = ", VAR.sizes(), \
                      ", strides() = ", VAR.strides()); \
    } \
  }

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
  // AccumulateGrad tries to stash strided (non-sparse) grads with memory layout
  // (strides) such that variables and grads interact efficiently in later
  // optimizer kernels, and grads interact efficiently with c10d::Reducer.cpp.
  //
  // Specifically, AccumulateGrad tries to ensure the following
  // (cf torch/csrc/autograd/utils/grad_layout_contract.h):
  //   (1) if variable.is_non_overlapping_and_dense(), the stashed grad's
  //       strides match variable.
  //   (2) else, stashed grad is rowmajor contiguous.
  // If variable's grad does not exist (!variable_grad.defined())
  // AccumulateGrad steals new_grad if it's stealable and obeys the contract
  // already, otherwise it deep copies new_grad into an obedient clone.
  //
  // If variable's grad already exists (variable_grad.defined()), new_grad must
  // be added to variable_grad.  If we aren't setting up for double backward
  // (!GradMode::is_enabled()), AccumulateGrad performs "variable_grad += new_grad"
  // in-place, which keeps variable_grad's layout. We assume (hope) variable_grad
  // was created obeying (1) or (2) at some point in the past.
  //
  // If we are setting up for double backward, AccumulateGrad updates the grad
  // out-of-place via "variable_grad + new_grad."  TensorIterator operator+ decides
  // result's layout.  Typically TensorIterator matches strides of the first arg,
  // so we once again assume (hope) variable_grad was originally created obeying
  // (1) or (2).
  //
  // AccumulateGrad does not enforce the contract with 100% certainty.  Examples:
  //  - If a user manually permutes a param or its grad, then runs a fwd+bwd,
  //    variable_grad += new_grad keeps variable_grad's layout without rechecking
  //    the contract.
  //  - If TensorIterator changes its corner cases about operator+'s result
  //    (for example, giving more or less priority to channels_last inputs, see
  //    https://github.com/pytorch/pytorch/pull/37968) the result may not obey.
  //
  // Fortunately, if a given grad doesn't satisfy (1) or (2), the penalty is
  // degraded performance in Reducer.cpp or optimizer kernels, not death by
  // assert or silently bad numerics.

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
      if (!GradMode::is_enabled() &&
          !new_grad.is_sparse() &&
          new_grad.use_count() <= num_expected_refs &&
          (new_grad.is_mkldnn() || utils::obeys_layout_contract(new_grad, variable))) {
        // we aren't setting up for double-backward
        // not sparse
        // no other user-visible tensor references new_grad
        // new_grad obeys the "Gradient Layout Contract", there has a special case,
        // For MKLDNN tensor, which is a opaque tensor, assuming it obeys layout_contract.
        // Under these conditions, we can steal new_grad without a deep copy.
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
          if (new_grad.is_mkldnn()) {
            update_grad(new_grad.clone());
          } else {
            // Deep copies new_grad according to the "Gradient Layout Contract."
            update_grad(utils::clone_obey_contract(new_grad, variable));
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
        auto result = new_grad + variable_grad;
        CHECK_RESULT(result, variable);
        update_grad(std::move(result));
      } else if (!at::inplaceIsVmapCompatible(variable_grad, new_grad)) {
        // Ideally we'd perform an in-place operation to avoid changing
        // the grad tensor. However, if that's impossible because the grads
        // are vmap-incompatible (See NOTE: [vmap-incompatible in-place operations]),
        // then we just add them out-of-place.
        auto result = variable_grad + new_grad;
        CHECK_RESULT(result, variable);
        update_grad(std::move(result));
      } else {
        // In this case we can avoid changing the grad tensor. There are three
        // scenarios when we'll hit this case:
        //
        // 1. `variable_grad` is sparse, and `new_grad` is sparse.
        // 2. `variable_grad` is dense, and `new_grad` is sparse.
        // 3. `variable_grad` is dense, and `new_grad` is dense.
        // 4. `variable_grad` is mkldnn, and `new_grad` is mkldnn.
        //
        // In all of these four cases, `variable_grad += new_grad` is a
        // valid operation which adds `new_grad` to `variable_grad` in
        // place. `variable_grad` is thus still referring to the same tensor
        // after the operation.
        // Also DistributedDataParallel(DDP) package relies on grad being
        // mutated in place for saving peak memory usage. DDP will still
        // work correctly if it is mutated out of place here, but DDP will
        // maintain one extra copy of grad tensors in buffer and thus
        // increase peak memory usage.
        variable_grad += new_grad;
        CHECK_RESULT(variable_grad, variable);
        // ^ We could enforce the contract more aggressively here by writing:
        // if (variable_grad.is_sparse() || new_grad.is_sparse()) {
        //   variable_grad += new_grad;
        // } else if (obeys_layout_contract(variable_grad, variable)) {
        //   variable_grad += new_grad;
        // } else {
        //   result = at::empty_strided(variable.sizes(), variable.strides(),
        //                              variable.options().memory_format(c10::nullopt));
        //   update_grad(at::native::add_out(result, variable_grad, new_grad, 1.0);
        // }
        // However, that accumulation is sometimes in place and sometimes not,
        // which may break user code.
      }
    } else {
      at::Tensor result;
      if (variable_grad.is_sparse() && !new_grad.is_sparse()) {
        // CPU backend throws an error on sparse + dense, so prefer dense + sparse here.
        result = new_grad + variable_grad;
      } else {
        // Assumes operator+ result typically matches strides of first arg,
        // and hopes variable_grad was originally created obeying layout contract.
        result = variable_grad + new_grad;
      }
      CHECK_RESULT(result, variable);
      update_grad(std::move(result));
      // ^ We could enforce the contract more aggressively here by saying
      // if (obeys_layout_contract(new_grad, variable)) {
      //   update_grad(new_grad + variable_grad);
      // } else {
      //   update_grad(variable_grad + new_grad);
      // }
      // such that the stashed grad is likely to have the right strides if
      // either variable_grad or new_grad already has the right strides.
      // We could enforce the contract with certainty by saying
      // auto result = variable_grad + new_grad (or vice versa), checking result's
      // layout, and copying to an obedient clone if necessary before update_grad.
      // The copy would require another gmem pass.  We can't create empty result with
      // the right layout then add_out into it with a single kernel, because GradMode
      // is enabled in this branch, and add_out isn't differentiable.
      // Maybe more trouble than it's worth.
    }
  }

  Variable variable;
};

#undef CHECK_RESULT

} // namespace autograd
} // namespace torch
