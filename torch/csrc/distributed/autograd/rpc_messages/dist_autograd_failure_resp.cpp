#include <torch/csrc/distributed/autograd/rpc_messages/dist_autograd_failure_resp.h>

namespace torch {
namespace distributed {
namespace autograd {

rpc::Message DistAutogradFailureResp::toMessageImpl() && {
  std::vector<torch::Tensor> tensors;
  std::vector<char> payload;
  return rpc::Message(
      std::move(payload),
      std::move(tensors),
      rpc::MessageType::DIST_AUTOGRAD_FAILURE_RESP);
}

std::unique_ptr<DistAutogradFailureResp> DistAutogradFailureResp::fromMessage(
    const rpc::Message& message /* unused */) {
  return std::unique_ptr<DistAutogradFailureResp>();
}

} // namespace autograd
} // namespace distributed
} // namespace torch
