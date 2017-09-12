#include "accumulate_grad.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/utils/auto_gpu.h"

using at::Tensor;

namespace torch { namespace autograd {

AccumulateGrad::AccumulateGrad(Variable variable_)
   : variable(std::move(variable_)) {
  num_inputs = 1;
  is_executable = 1;
  input_sizes.emplace_back(variable_);
}

auto AccumulateGrad::apply(const variable_list& grads) -> variable_list {
  // XXX: this method is not thread-safe!
  check_input_variables("AccumulateGrad", grads, 1, 0);

  if (!grads[0].defined())
    return {};
  if (variable.grad_fn())
    throw std::logic_error("leaf variable has been moved into the graph interior");
  if (variable.current_version() != 0)
    throw std::runtime_error("leaf variable was used in an inplace operation");
  if (!variable.requires_grad())
    return {};

  auto new_grad = grads[0];
  for (auto& hook : variable.hooks()) {
    new_grad = (*hook)({new_grad})[0];
  }

  // TODO: Currently if var.grad is volatile and new-grad is non-volatile we
  // accumulate in-place. We should reconsider this and perhaps add the
  // gradients out-of-place.

  auto& grad = variable.grad();
  if (!grad.defined()) {
    grad = apply_fn<Clone>()(new_grad);
  } else if (grad.is_volatile()) {
    // This case is not strictly necessary, but it makes the first-order only case
    // slightly more efficient and, what's more important, more predictable for
    // the users. Thanks to this case we can avoid changing the grad tensor,
    // a thing never promised and documented, but used in some hacks seen
    // on the internet.
    AutoGPU guard(grad);
    if (grad.type().isSparse() && !new_grad.type().isSparse()) {
      grad.data() = new_grad.data() + grad.data();
    } else {
      grad.data() += new_grad.data();
    }
  } else {
    // If grad is non-volatile, it should stay like that
    if (new_grad.is_volatile()) {
      new_grad = make_variable(new_grad.data());
    }
    variable.grad() = apply_fn<Add>()(grad, new_grad);
  }

  return variable_list();
};

}} // namespace torch::autograd
