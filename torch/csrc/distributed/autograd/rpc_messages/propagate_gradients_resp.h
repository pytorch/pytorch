#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace autograd {

// Response for the PropagateGradients call. Currently, this class is mostly
// just a placeholder and sends an empty message over the wire. The purpose of
// this RPC command is to indicate whether or not the PropagateGradientsReq call
// was successfully or not.
class TORCH_API PropagateGradientsResp : public rpc::RpcCommandBase {
 public:
  PropagateGradientsResp() = default;
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  static std::unique_ptr<PropagateGradientsResp> fromMessage(
      const rpc::Message& message);
};

} // namespace autograd
} // namespace distributed
} // namespace torch
