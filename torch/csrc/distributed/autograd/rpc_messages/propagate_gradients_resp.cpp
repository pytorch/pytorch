#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.h>

namespace torch {
namespace distributed {
namespace autograd {

rpc::Message PropagateGradientsResp::toMessageImpl() && {
  return rpc::Message({}, {}, rpc::MessageType::BACKWARD_AUTOGRAD_RESP);
}

std::unique_ptr<PropagateGradientsResp> PropagateGradientsResp::fromMessage(
    const rpc::Message& message) {
  return std::unique_ptr<PropagateGradientsResp>();
}

} // namespace autograd
} // namespace distributed
} // namespace torch
