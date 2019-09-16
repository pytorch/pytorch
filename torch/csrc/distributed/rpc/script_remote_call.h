#pragma once

#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/pickler.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

using torch::jit::Operator;

// A ScriptCall instance represents an invocation of a builtin operator for a
// TorchScript function (not implemented yet). If it is a builtin operator, it
// contains a shared ptr to the `Operator` and a list of arguments.
class TORCH_API ScriptRemoteCall final : public ScriptCall {
 public:
  ScriptRemoteCall(
      std::shared_ptr<Operator> op,
      std::vector<at::IValue>&& args,
      at::IValue retRRefId,
      at::IValue retForkId);

  const at::IValue& retRRefId();
  const at::IValue& retForkId();

  Message toMessage() override;
  static std::unique_ptr<ScriptRemoteCall> fromMessage(const Message& message);

 private:
  const at::IValue retRRefId_;
  const at::IValue retForkId_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
