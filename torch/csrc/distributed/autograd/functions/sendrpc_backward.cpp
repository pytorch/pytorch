#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>

namespace torch {
namespace distributed {
namespace autograd {

torch::autograd::variable_list SendRpcBackward::apply(
    torch::autograd::variable_list&& inputs) {
  TORCH_INTERNAL_ASSERT(
      inputs.empty(), "SendRpcBackward should receive no inputs");

  // Each grad variable should be valid!
  for (const auto& grad : grads_) {
    TORCH_INTERNAL_ASSERT(
        grad.defined(), "BUG!: SendRpcBackward didn't receive valid gradients");
  }

  // Simply forwards the gradients over.
  return std::move(grads_);
}

void SendRpcBackward::setGrads(const torch::autograd::variable_list& grads) {
  grads_ = grads;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
