#include <torch/csrc/autograd/functions/accumulate_grad.h>

#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/tensor.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/dynamo/compiled_autograd.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace torch {
namespace autograd {

// AccumulateGrad sets sequence_nr to the max value so it's always called
// ASAP during backwards.
AccumulateGrad::AccumulateGrad(Variable variable_)
    : Node(/*sequence_nr=*/UINT64_MAX), variable(std::move(variable_)) {
  add_input_metadata(variable);
}

auto AccumulateGrad::apply(variable_list&& grads) -> variable_list {
  check_input_variables("AccumulateGrad", grads, 1, 0);

  if (!grads[0].defined())
    return {};
  if (variable.grad_fn())
    throw std::logic_error(
        "leaf variable has been moved into the graph interior");
  if (!variable.requires_grad())
    return {};

  // std::move(grads[0]) to avoid bumping up refcount
  at::Tensor new_grad = std::move(grads[0]);

  // Acquire lock to here protect thread safety on variable, this ensures
  // AccumulateGrad does not race to shared variable from different threads
  // when updating the gradients. We don't ensure thread safety on hooks
  // and rely on user to provide thread safe hooks
  // see Note [Thread Safety on Autograd Node]
  std::lock_guard<std::mutex> lock(mutex_);

  at::Tensor& grad = variable.mutable_grad();

  // If the function has post hooks (for example, a DDP allreduce hook),
  // call_function in Engine.cpp will temporarily bump the expected refcount
  // by one, hence the addition of !post_hooks().empty() for 'num_expected_refs'
  // in addition to the one reference that we're holding.
  // 'num_expected_refs' is used to determine whether or not we should clone
  // the grad or can steal the grad.
  accumulateGrad(
      variable,
      grad,
      new_grad,
      1 + !post_hooks().empty() /* num_expected_refs */,
      [&grad](at::Tensor&& grad_update) { grad = std::move(grad_update); });

  return variable_list();
}

void AccumulateGrad::compiled_args(CompiledNodeArgs& args) {
  if (args.cond(variable.defined() && variable.requires_grad())) {
    args.collect(variable);
    args.collect(variable.grad());
    args.set_grad_target(variable);
  }
}
variable_list AccumulateGrad::apply_with_saved(
    const variable_list& grads,
    SwapSavedVariables& saved) {
  if (!(variable.defined() && variable.requires_grad())) {
    return variable_list();
  }
  TORCH_INTERNAL_ASSERT(!variable.grad_fn() && grads.size() == 1);
  TORCH_CHECK(grads[0].defined(), "not implemented for compiled autograd")
  at::Tensor variable_copy = variable;
  at::Tensor grad_copy = variable.grad();
  saved.before(variable_copy);
  saved.before(grad_copy);
  int callback_count = 0;
  accumulateGrad(
      variable_copy,
      grad_copy,
      grads[0],
      0 /* num_expected_refs, 0 disables in-place update */,
      [&callback_count, &saved](const at::Tensor& grad_update) {
        callback_count++;
        saved.set_grad_value(grad_update);
      });
  TORCH_INTERNAL_ASSERT(callback_count == 1);
  saved.after(variable_copy);
  saved.after(grad_copy);
  return variable_list();
}

} // namespace autograd
} // namespace torch
