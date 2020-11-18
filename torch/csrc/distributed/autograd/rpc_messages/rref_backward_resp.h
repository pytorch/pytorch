#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace autograd {

// Response for the RRefBackwardReq.
class TORCH_API RRefBackwardResp : public rpc::RpcCommandBase {
 public:
  RRefBackwardResp() = default;
  rpc::Message toMessageImpl() && override;
  static std::unique_ptr<RRefBackwardResp> fromMessage(
      const rpc::Message& message);
};

} // namespace autograd
} // namespace distributed
} // namespace torch
