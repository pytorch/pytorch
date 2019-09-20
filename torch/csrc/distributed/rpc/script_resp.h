#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/jit/pickler.h>

namespace torch {
namespace distributed {
namespace rpc {

// Return value of a builtin operator or a TorchScript function.
class TORCH_API ScriptResp final : public RpcCommandBase {
 public:
  explicit ScriptResp(at::IValue&& values);

  const at::IValue& value();
  Message toMessage() override;
  static std::unique_ptr<ScriptResp> fromMessage(const Message& message);

 private:
  const at::IValue value_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
