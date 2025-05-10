#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch::distributed::autograd {

// Response for the RRefBackwardReq.
class TORCH_API RRefBackwardResp : public rpc::RpcCommandBase {
 public:
  RRefBackwardResp() = default;
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  static std::unique_ptr<RRefBackwardResp> fromMessage(
      const rpc::Message& message);
};

} // namespace torch::distributed::autograd
