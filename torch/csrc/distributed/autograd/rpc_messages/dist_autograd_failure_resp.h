#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <vector>

namespace torch {
namespace distributed {
namespace autograd {

// Empty response for DistAutogradFailureReq. Send to acknowledge receipt of
// a DistAutogradFailureReq.
class TORCH_API DistAutogradFailureResp : public rpc::RpcCommandBase {
 public:
  DistAutogradFailureResp() = default;
  // Serialization and deserialization methods.
  rpc::Message toMessageImpl() && override;
  static std::unique_ptr<DistAutogradFailureResp> fromMessage(
      const rpc::Message& message);
};

} // namespace autograd
} // namespace distributed
} // namespace torch
