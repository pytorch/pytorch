#include <torch/csrc/distributed/autograd/Utils.h>
#include <torch/csrc/autograd/functions/utils.h>

namespace torch {
namespace distributed {
namespace autograd {

std::shared_ptr<SendRpcBackwards> addSendRpcBackward(
    at::ArrayRef<c10::IValue> ivalues) {
  // Extract the tensors.
  std::vector<at::Tensor> tensors;
  for (const auto& ivalue : ivalues) {
    if (ivalue.isTensor()) {
      tensors.push_back(ivalue.toTensor());
    }
  }

  std::shared_ptr<SendRpcBackwards> grad_fn;
  if (torch::autograd::compute_requires_grad(tensors)) {
    grad_fn = std::make_shared<SendRpcBackwards>();
    grad_fn->set_next_edges(torch::autograd::collect_next_edges(tensors));
  }
  return grad_fn;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
