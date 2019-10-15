#pragma once

#include <torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <vector>

namespace torch {
namespace distributed {
namespace autograd {

// Used to request other workers to clean up their autograd context.
class TORCH_API CleanupAutogradContextReq : public rpc::RpcCommandBase {
 public:
  CleanupAutogradContextReq();
  // Serialization and deserialization methods.
  rpc::Message toMessage() && override;
  static std::unique_ptr<CleanupAutogradContextReq> fromMessage(
      const rpc::Message& message); // todo
};

} // namespace autograd
} // namespace distributed
} // namespace torch
