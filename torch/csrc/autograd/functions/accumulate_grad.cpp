#include "torch/csrc/autograd/functions/accumulate_grad.h"

#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/utils/auto_gpu.h"

using at::Tensor;

namespace torch { namespace autograd {

AccumulateGrad::AccumulateGrad(Variable variable_)
    : Function(/*num_inputs=*/1), variable(std::move(variable_)) {}

auto AccumulateGrad::apply(const variable_list& grads) -> variable_list {
  // XXX: this method is not thread-safe!
  check_input_variables("AccumulateGrad", grads, 1, 0);

  if (!grads[0].defined())
    return {};
  if (variable.grad_fn())
    throw std::logic_error("leaf variable has been moved into the graph interior");
  if (!variable.requires_grad())
    return {};

  auto new_grad = grads[0];
  for (auto& hook : variable.hooks()) {
    new_grad = (*hook)({new_grad})[0];
  }

  at::Tensor& grad = variable.grad();
  if (!grad.defined()) {
    variable.grad() = new_grad.clone();
  } else if (!GradMode::is_enabled()) {
    Variable& grad_variable = as_variable_ref(grad);
    // This case is not strictly necessary, but it makes the first-order only case
    // slightly more efficient and, what's more important, more predictable for
    // the users. Thanks to this case we can avoid changing the grad tensor,
    // a thing never promised and documented, but used in some hacks seen
    // on the internet.
    if (grad_variable.type().is_sparse() && !new_grad.type().is_sparse()) {
      grad_variable.data() = new_grad.data() + grad_variable.data();
    } else {
      grad_variable.data() += new_grad.data();
    }
  } else {
    variable.grad() = grad + new_grad;
  }

  return variable_list();
}

}} // namespace torch::autograd
