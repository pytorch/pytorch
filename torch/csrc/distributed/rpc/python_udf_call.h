#pragma once

#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace rpc {

// RPC call representing calling a Python UDF over RPC.
class TORCH_API PythonUDFCall final : public RpcCommandBase {
 public:
  explicit PythonUDFCall(
      std::vector<char> pickledPayload,
      std::vector<torch::Tensor> tensors);

  Message toMessage() && override;

  static std::unique_ptr<PythonUDFCall> fromMessage(const Message& message);

  const std::vector<char>& pickledPayload() const;

  const std::vector<torch::Tensor>& tensors() const;

 private:
  std::vector<char> pickledPayload_;
  std::vector<torch::Tensor> tensors_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
