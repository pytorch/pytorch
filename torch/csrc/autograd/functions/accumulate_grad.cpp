#include <torch/csrc/autograd/functions/accumulate_grad.h>

#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/functions/utils.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

using at::Tensor;

namespace torch { namespace autograd {

// AccumulateGrad sets sequence_nr to the max value so it's always called
// ASAP during backwards.
AccumulateGrad::AccumulateGrad(Variable variable_)
    : Node(/*sequence_nr=*/UINT64_MAX)
    , variable(std::move(variable_)) {
  add_input_metadata(variable);
}

void AccumulateGrad::accumulateGradAndCallHooks(
    const Variable& variable,
    at::Tensor variable_grad,
    const at::Tensor& new_grad,
    size_t num_expected_refs,
    const std::function<void(at::Tensor&&)>& update_grad) {
  // Copy since we need to work with non-const Tensor. Grab the original
  // use_count beforehand though.
  size_t new_grad_use_count = new_grad.use_count();
  at::Tensor new_grad_copy = new_grad;

  for (auto& hook : impl::hooks(variable)) {
    new_grad_copy = (*hook)({new_grad_copy})[0];
  }

  if (!variable_grad.defined()) {
    // under following condition, we can avoid clone()
    if (!GradMode::is_enabled() && !new_grad_copy.is_sparse() &&
        new_grad_copy.is_contiguous() &&
        new_grad_use_count <= num_expected_refs) {
      // first check it is in first-order grad only mode
      // then check not sparse before is_contiguous
      // then check contiguous, otherwise later in place accumulation may fail
      // and lastly, check if the use_count is less than or equal to the number
      // of references we expect before grabbing it. The number of references we
      // expect is basically internal structures that are holding references to
      // the Tensor and that is fine since these are not exposed to the user.
      update_grad(new_grad_copy.detach());
    } else {
      if (new_grad_copy.is_sparse()) {
        update_grad(new_grad_copy.clone());
      } else {
        update_grad(new_grad_copy.clone(at::MemoryFormat::Contiguous));
      }
    }
  } else if (!GradMode::is_enabled()) {
    // This case is not strictly necessary, but it makes the first-order only case
    // slightly more efficient.
    if (variable_grad.is_sparse() && !new_grad_copy.is_sparse()) {
      // If `grad_variable` is sparse and `new_grad_copy` is not sparse, their
      // sum is not sparse, and we must change the TensorImpl type of
      // `grad_variable` for it to store the result. However, changing the
      // TensorImpl type of a tensor requires changing the tensor itself, and
      // thus in this case we have to change the grad tensor.
      update_grad(new_grad_copy + variable_grad);
    } else {
      // In this case we can avoid changing the grad tensor. There are three
      // scenarios when we'll hit this case:
      //
      // 1. `grad_variable` is sparse, and `new_grad_copy` is sparse.
      // 2. `grad_variable` is dense, and `new_grad_copy` is sparse.
      // 3. `grad_variable` is dense, and `new_grad_copy` is dense.
      //
      // In all of these three cases, `grad_variable += new_grad_copy` is a
      // valid operation which adds `new_grad_copy` to `grad_variable` in place.
      // `grad_variable` is thus still referring to the same tensor after the
      // operation.
      variable_grad += new_grad_copy;
    }
  } else {
    update_grad(variable_grad + new_grad_copy);
  }
}

auto AccumulateGrad::apply(variable_list&& grads) -> variable_list {
  // XXX: this method is not thread-safe!
  check_input_variables("AccumulateGrad", grads, 1, 0);

  if (!grads[0].defined())
    return {};
  if (variable.grad_fn())
    throw std::logic_error(
        "leaf variable has been moved into the graph interior");
  if (!variable.requires_grad())
    return {};

  const auto& new_grad = grads[0];
  at::Tensor& grad = variable.grad();
  // If the function has post hooks (for example, a DDP allreduce hook),
  // call_function in Engine.cpp will temporarily bump the refcount by one,
  // hence the addition of !post_hooks().empty() for num_expected_refs in
  // addition to the one reference that we're holding.
  accumulateGradAndCallHooks(
      variable,
      grad,
      new_grad,
      1 + !post_hooks().empty() /* num_expected_refs */,
      [&grad](at::Tensor&& grad_update) { grad = grad_update; });

  return variable_list();
}
}} // namespace torch::autograd
