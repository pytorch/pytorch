#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_resp.h>

namespace torch {
namespace distributed {
namespace autograd {

rpc::Message RRefBackwardResp::toMessageImpl() && {
  return rpc::Message({}, {}, rpc::MessageType::RREF_BACKWARD_RESP);
}

std::unique_ptr<RRefBackwardResp> RRefBackwardResp::fromMessage(
    const rpc::Message& message) {
  TORCH_INTERNAL_ASSERT(message.type() == rpc::MessageType::RREF_BACKWARD_RESP);
  return std::unique_ptr<RRefBackwardResp>();
}

} // namespace autograd
} // namespace distributed
} // namespace torch
