#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>

namespace torch {
namespace distributed {
namespace autograd {

torch::autograd::variable_list SendRpcBackward::apply(
    torch::autograd::variable_list&& grads) {
  // Each grad variable should be valid!
  for (const auto& grad : grads) {
    TORCH_CHECK(
        grad.defined(), "BUG!: SendRpcBackward didn't receive valid gradients");
  }

  // Simply forwards the gradients over.
  // TODO: Improve this as we build out more parts of distributed autograd.
  return std::move(grads);
}

} // namespace autograd
} // namespace distributed
} // namespace torch
