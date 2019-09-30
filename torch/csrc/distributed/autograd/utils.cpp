#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/distributed/autograd/utils.h>

namespace torch {
namespace distributed {
namespace autograd {

std::shared_ptr<SendRpcBackward> addSendRpcBackward(
    const std::vector<torch::Tensor>& tensors) {
  // Attach the appropriate autograd edges.
  std::shared_ptr<SendRpcBackward> grad_fn;
  if (torch::autograd::compute_requires_grad(tensors)) {
    grad_fn = std::make_shared<SendRpcBackward>();
    grad_fn->set_next_edges(torch::autograd::collect_next_edges(tensors));

    // Add the appropriate input metadata for the grad_fn.
    for (const auto& tensor : tensors) {
      grad_fn->add_input_metadata(tensor);
    }
  }
  return grad_fn;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
