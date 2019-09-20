#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/pickler.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

// Temporary solution of RRef operations.
// TODO: Remove all these messages and use rpc + registered functions instead.
class TORCH_API RRefMessageBase : public RpcCommandBase {
 public:
  RRefMessageBase(at::IValue value, MessageType type)
      : value_(std::move(value)), type_(type) {}

  const at::IValue& value();
  at::IValue& valueRef();

  Message toMessage() override;
  static at::IValue fromMessage(const Message& message);

 private:
  at::IValue value_;
  const MessageType type_;
};

// UserRRef uses this message to fetch the remote RRef value from the owner.
class TORCH_API ScriptRRefFetchCall final : public RRefMessageBase {
 public:
  ScriptRRefFetchCall(at::IValue rrefForkData)
      : RRefMessageBase(std::move(rrefForkData), MessageType::RREF_FETCH_CALL) {
  }

  static std::unique_ptr<ScriptRRefFetchCall> fromMessage(
      const Message& message);
};

// OwnerRRef uses this message to send the RRef value to a remote UserRRef
class TORCH_API ScriptRRefFetchRet final : public RRefMessageBase {
 public:
  ScriptRRefFetchRet(at::IValue value)
      : RRefMessageBase(std::move(value), MessageType::RREF_FETCH_RET) {}

  static ScriptRRefFetchRet fromMessage(const Message& message);
};

// Creator UserRRef uses this message to notify OwnerRRef on create.
class TORCH_API ScriptRRefCreate final : public RRefMessageBase {
 public:
  ScriptRRefCreate(at::IValue value)
      : RRefMessageBase(std::move(value), MessageType::RREF_USER_CREATE) {}

  static std::unique_ptr<ScriptRRefCreate> fromMessage(const Message& message);
};

// UserRRef (regardless of it's the creator or not) uses this message to notify
// OwnerRRef on delete.
class TORCH_API ScriptRRefDelete final : public RRefMessageBase {
 public:
  ScriptRRefDelete(at::IValue value)
      : RRefMessageBase(std::move(value), MessageType::RREF_USER_DELETE) {}

  static std::unique_ptr<ScriptRRefDelete> fromMessage(const Message& message);
};

} // namespace rpc
} // namespace distributed
} // namespace torch
