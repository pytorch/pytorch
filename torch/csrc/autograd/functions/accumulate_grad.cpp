#include "accumulate_grad.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/utils/auto_gpu.h"

namespace torch { namespace autograd {

AccumulateGrad::AccumulateGrad(std::shared_ptr<Variable> _variable)
    : variable(_variable)
    , variable_grad(_variable->grad) {
  is_executable = _variable->requires_grad;
  num_inputs = 1;
}

auto AccumulateGrad::acc_inplace(std::shared_ptr<Variable>& grad,
    std::shared_ptr<Variable>& new_grad) -> void {
  auto& grad_data = grad->data;
  auto& new_grad_data = new_grad->data;
  AutoGPU guard(grad_data);

  if (grad_data.type().isSparse() && !new_grad_data.type().isSparse()) {
    grad->data = new_grad_data + grad_data;
  } else {
    grad_data += new_grad_data;
  }
}

auto AccumulateGrad::apply(const variable_list& grads) -> variable_list {
  // XXX: this method is not thread-safe!
  check_input_variables("AccumulateGrad", grads, 1, 0);
  auto new_grad = grads[0];

  if (!new_grad) return {};

  auto var = variable.lock();
  // It's possible that the Variable went out of scope and was freed.
  // We still need to handle the unlikely case of someone holding to its grad.
  if (!var) {
    auto var_grad = variable_grad.lock();
    // Everything was freed. Nothing to do.
    if (!var_grad) return variable_list();
    // Now here's the hard part. If both the new_grad and var_grad are volatile
    // then we just acumulate the data in place (as we'd do if the Variable was
    // alive). Otherwise, we'd need to perform the out-of-place reduction, but
    // since the user only holds a reference to .grad and there's no way to
    // give him the new Value, we just assume that they know these attributes
    // are changing when using higher order graphs.
    if (!var_grad->is_volatile || !new_grad->is_volatile) return variable_list();
    acc_inplace(var_grad, new_grad);
    return variable_list();
  }

  if (var->grad_fn)
    throw std::logic_error("leaf variable has been moved into the graph interior");
  if (**var->version_counter != 0)
    throw std::runtime_error("leaf variable was used in an inplace operation");
  if (var->get_grad_accumulator().get() != this)
    throw std::logic_error("AccumulateGrad's variable is not bound to it");

  for (auto& hook : var->hooks) {
    new_grad = (*hook)({new_grad})[0];
  }

  if (!var->grad) {
    var->grad = Clone().apply({new_grad})[0];
    variable_grad = var->grad; // We need to update our reference
  // This case is not strictly necessary, but it makes the first-order only case
  // slightly more efficient and, what's more important, more predictable for
  // the users. Thanks to this case we can avoid changing the grad tensor,
  // a thing never promised and documented, but used in some hacks seen
  // on the internet.
  } else if (var->grad->is_volatile) {
    acc_inplace(var->grad, new_grad);
  } else {
    // Once the grad becomes not volatile, it should stay like that
    if (!var->grad->is_volatile && new_grad->is_volatile) {
      new_grad = std::make_shared<Variable>(new_grad->data, false, false);
    }
    var->grad = Add().apply({var->grad, new_grad})[0];
  }

  return variable_list();
};

}} // namespace torch::autograd
