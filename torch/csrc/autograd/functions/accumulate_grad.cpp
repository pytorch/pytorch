#include "torch/csrc/autograd/functions/accumulate_grad.h"

#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/functions/utils.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

using at::Tensor;

namespace torch { namespace autograd {

// AccumulateGrad sets sequence_nr to the max value so it's always called
// ASAP during backwards.
AccumulateGrad::AccumulateGrad(Variable variable_)
    : Function(/*sequence_nr=*/UINT64_MAX)
    , variable(std::move(variable_)) {
  add_input_metadata(variable);
}

auto AccumulateGrad::apply(variable_list&& grads) -> variable_list {
  // XXX: this method is not thread-safe!
  check_input_variables("AccumulateGrad", grads, 1, 0);

  if (!grads[0].defined())
    return {};
  if (variable.grad_fn())
    throw std::logic_error("leaf variable has been moved into the graph interior");
  if (!variable.requires_grad())
    return {};

  auto new_grad = std::move(grads[0]);
  for (auto& hook : variable.hooks()) {
    new_grad = (*hook)({new_grad})[0];
  }

  at::Tensor& grad = variable.grad();
  if (!grad.defined()) {
    // under following condition, we can avoid clone()
    if (!GradMode::is_enabled()
        && !new_grad.is_sparse()
        && new_grad.is_contiguous()
        && new_grad.use_count() == 1) {
      // first check it is in first-order grad only mode
      // then check not sparse before is_contiguous
      // then check contiguous, otherwise later in place accumulation may fail
      // and lastly, check it is the last reference before we grab it
      variable.grad() = new_grad.detach();
    } else {
      variable.grad() = new_grad.clone();
    }
  } else if (!GradMode::is_enabled()) {
    Variable& grad_variable = as_variable_ref(grad);
    // This case is not strictly necessary, but it makes the first-order only case
    // slightly more efficient and, what's more important, more predictable for
    // the users. Thanks to this case we can avoid changing the grad tensor,
    // a thing never promised and documented, but used in some hacks seen
    // on the internet.
    if (grad_variable.is_sparse() && !new_grad.is_sparse()) {
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
