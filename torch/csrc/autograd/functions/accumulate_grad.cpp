#include "accumulate_grad.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/utils/auto_gpu.h"

namespace torch { namespace autograd {

AccumulateGrad::AccumulateGrad(std::shared_ptr<Variable> _variable)
    : variable(std::move(_variable)) {
  is_executable = variable->requires_grad;
  num_inputs = 1;
}

auto AccumulateGrad::apply(const variable_list& _grads) -> variable_list {
  // XXX: this method is not thread-safe!
  if (_grads.size() != 1) throw std::runtime_error("AccumulateGrad expects exactly 1 input");
  auto& var = *variable;
  auto grads = _grads;

  if (var.grad_fn)
    throw std::logic_error("leaf variable has been moved into the graph interior");
  if (**var.version_counter != 0)
    throw std::runtime_error("leaf variable was used in an inplace operation");
  if (var.get_grad_accumulator().get() != this)
    throw std::runtime_error("AccumulateGrad's variable is not bound to it");

  auto new_grad = grads[0];
  for (auto& hook : var.hooks) {
    new_grad = (*hook)({new_grad})[0];
  }

  if (!var.grad) {
    auto clone_fn = std::make_shared<Clone>();
    var.grad = clone_fn->apply({new_grad})[0];
  // This case is not strictly necessary, but it makes the first-order only case
  // slightly more efficient and, what's more important, more predictable for
  // the users. Thanks to this case we can avoid changing the grad tensor,
  // a thing never promised and documented, but used in some hacks seen
  // on the internet.
  } else if (var.grad->is_volatile) {
    auto& grad_data = var.grad->data;
    auto& new_grad_data = new_grad->data;
    AutoGPU guard(grad_data->getDevice());

    // The grad may need a promotion from a sparse to dense type
    if (grad_data->isSparse() && !new_grad_data->isSparse()) {
      std::unique_ptr<thpp::Tensor> result = new_grad_data->newTensor();
      result->cadd(*new_grad_data, *grad_data);
      var.grad = Variable::of(std::move(result), true);
    } else {
      grad_data->cadd(*grad_data, *new_grad_data);
    }
  } else {
    auto add_fn = std::make_shared<Add>();
    // Once the grad becomes not volatile, it should stay like that
    if (!var.grad->is_volatile && new_grad->is_volatile) {
      new_grad = std::make_shared<Variable>(
              std::unique_ptr<thpp::Tensor>(new_grad->data->clone_shallow()), false, false);
    }
    var.grad = add_fn->apply({var.grad, new_grad})[0];
  }

  return variable_list();
};

}} // namespace torch::autograd

