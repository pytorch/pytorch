#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <vector>

namespace torch {
namespace distributed {
namespace autograd {

// Used to notify other workers when there is an autograd error.
class TORCH_API DistAutogradFailureReq : public rpc::RpcCommandBase {
 public:
  DistAutogradFailureReq(int64_t context_id, std::string errorMsg);
  // Serialization and deserialization methods.
  rpc::Message toMessageImpl() && override;
  static std::unique_ptr<DistAutogradFailureReq> fromMessage(
      const rpc::Message& message);

  int64_t getContextId();
  std::string getErrorMsg();

 private:
  int64_t context_id_;
  std::string errorMsg_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
