#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace autograd {

// Empty response for CleanupAutogradContextReq. Send to acknowledge receipt of
// a CleanupAutogradContextReq.
class TORCH_API CleanupAutogradContextResp : public rpc::RpcCommandBase {
 public:
  CleanupAutogradContextResp() = default;
  // Serialization and deserialization methods.
  rpc::Message toMessageImpl() && override;
  static std::unique_ptr<CleanupAutogradContextResp> fromMessage(
      const rpc::Message& message);
};

} // namespace autograd
} // namespace distributed
} // namespace torch
