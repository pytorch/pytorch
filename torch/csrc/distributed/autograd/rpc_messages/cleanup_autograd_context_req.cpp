#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>

namespace torch {
namespace distributed {
namespace autograd {

CleanupAutogradContextReq::CleanupAutogradContextReq(){

};

rpc::Message CleanupAutogradContextReq::toMessage() && {
  std::vector<torch::Tensor> tensors;
  std::vector<char> payload;
  return rpc::Message(
      std::move(payload),
      std::move(tensors),
      rpc::MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ);
}

std::unique_ptr<CleanupAutogradContextReq> CleanupAutogradContextReq::
    fromMessage(const rpc::Message& message) {
  return std::unique_ptr<CleanupAutogradContextReq>(
      new CleanupAutogradContextReq()); // todo
}

} // namespace autograd
} // namespace distributed
} // namespace torch
