#pragma once

#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <vector>

namespace torch::distributed::rpc {

using torch::jit::Operator;

// A ScriptRemoteCall instance represents an invocation of `dist.remote` on a
// builtin operator. Currently, it does not support using RRef as arguments yet.
// Besides the operator and a vector of arguments, ScriptRemoteCall also
// contains the RRefId and the ForkId of the return value RRef.
class TORCH_API ScriptRemoteCall final : public ScriptCall {
 public:
  // Constructor for builtin operator call.
  ScriptRemoteCall(
      std::shared_ptr<Operator> op,
      std::vector<at::IValue>&& stack,
      const RRefId& retRRefId,
      const ForkId& retForkId);

  // Constructor for TorchScript function call.
  ScriptRemoteCall(
      const c10::QualifiedName& qualifiedName,
      std::vector<at::IValue>&& stack,
      const RRefId& retRRefId,
      const ForkId& retForkId,
      const bool isAsyncExecution);

  inline const RRefId& retRRefId() const {
    return retRRefId_;
  }

  inline const ForkId& retForkId() const {
    return retForkId_;
  }

  static std::unique_ptr<ScriptRemoteCall> fromIValues(
      std::vector<at::IValue>& ivalues);

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<ScriptRemoteCall> fromMessage(const Message& message);

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const RRefId retRRefId_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const ForkId retForkId_;
};

} // namespace torch::distributed::rpc
