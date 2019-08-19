#include <torch/csrc/distributed/autograd/functions/SendRpcBackwards.h>

namespace torch {
namespace distributed {
namespace autograd {

torch::autograd::variable_list SendRpcBackwards::apply(
    torch::autograd::variable_list&& grads) {
  // Simply forwards the gradients over.
  // TODO: Improve this as we build out more parts of distributed autograd.
  return std::move(grads);
}

} // namespace autograd
} // namespace distributed
} // namespace torch
