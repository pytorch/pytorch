#pragma once

#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace rpc {

// RPC call representing the response of a Python UDF over RPC.
class TORCH_API PythonUDFResp final : public RpcCommandBase {
 public:
  PythonUDFResp(std::vector<char> pickledPayload);

  // Destructively creates a message to avoid copies.
  Message toMessage() override;

  static std::unique_ptr<PythonUDFResp> fromMessage(const Message& message);

  const std::vector<char>& pickledPayload() const;

 private:
  std::vector<char> pickledPayload_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
