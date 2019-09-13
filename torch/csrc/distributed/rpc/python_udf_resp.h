#pragma once

#include <torch/csrc/distributed/rpc/rpc_base.h>

namespace torch {
namespace distributed {
namespace rpc {

// RPC call representing the response of a Python UDF over RPC.
class PythonUDFResp final : public RpcBase {
 public:
  PythonUDFResp(std::vector<char>&& pickledPayload);

  PythonUDFResp(const std::vector<char>& pickledPayload);

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
