#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/pickler.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

// Temporary solution of RRef operations.
// TODO: Remove all these messages and use rpc + registered functions instead.
class TORCH_API RRefMessageBase {
 public:
  RRefMessageBase(at::IValue value, MessageType type)
      : value_(std::move(value)), type_(type) {}

  const at::IValue& value();
  at::IValue& valueRef();

  virtual Message toMessage() const;
  static at::IValue fromMessage(const Message& message);

 protected:
  at::IValue value_;
  const MessageType type_;
};

// UserRRef uses this message to fetch the remote RRef value from the owner.
class TORCH_API ScriptRRefFetchCall final : public RRefMessageBase {
 public:
  ScriptRRefFetchCall(at::IValue rrefForkData)
      : RRefMessageBase(
            std::move(rrefForkData),
            MessageType::SCRIPT_RREF_FETCH_CALL) {}

  static ScriptRRefFetchCall fromMessage(const Message& message);
};

class TORCH_API PythonRRefFetchCall final : public RRefMessageBase {
 public:
  PythonRRefFetchCall(at::IValue rrefForkData)
      : RRefMessageBase(
            std::move(rrefForkData),
            MessageType::PYTHON_RREF_FETCH_CALL) {}

  static PythonRRefFetchCall fromMessage(const Message& message);
};

// OwnerRRef uses this message to send the RRef value to a remote UserRRef
class TORCH_API ScriptRRefFetchRet final : public RRefMessageBase {
 public:
  ScriptRRefFetchRet(at::IValue value)
      : RRefMessageBase(std::move(value), MessageType::RREF_FETCH_RET) {}

  static ScriptRRefFetchRet fromMessage(const Message& message);
};

// UserRRef (regardless it's the creator or not) uses this message to notiify
// OwnerRRef on delete.
class TORCH_API ScriptUserDelete final : public RRefMessageBase {
 public:
  ScriptUserDelete(at::IValue value)
      : RRefMessageBase(std::move(value), MessageType::RREF_USER_DELETE) {}

  static ScriptUserDelete fromMessage(const Message& message);
};

// The OwnerRRef uses this message to accept a UserRRef. A UserRRef cannot be
// deleted before receiving this message.
class TORCH_API ScriptUserAccept final {
 public:
  ScriptUserAccept(
      worker_id_t owner, const RRefId& rrefId, const ForkId& forkId)
      : owner_(owner), rrefId_(rrefId), forkId_(forkId) {}

  Message toMessage();
  static ScriptUserAccept fromMessage(const Message& message);

  const worker_id_t owner_;
  const RRefId rrefId_;
  const ForkId forkId_;
};

// A UserRRef uses this message to notify owner on fork.
class TORCH_API ScriptForkNotify final {
 public:
  ScriptForkNotify(
      worker_id_t owner,
      const RRefId& rrefId,
      const ForkId& forkId,
      worker_id_t forkDst)
      : owner_(owner), rrefId_(rrefId), forkId_(forkId), forkDst_(forkDst) {}

  Message toMessage() const;
  static ScriptForkNotify fromMessage(const Message& message);

  const worker_id_t owner_;
  const RRefId rrefId_;
  const ForkId forkId_;
  const worker_id_t forkDst_;
};

// The OwnerRRef uses this message to a UserRRef that its fork request has been
// accepted. A UserRRef cannot be deleted if it has any pending fork requests.
class TORCH_API ScriptForkAccept final {
 public:
  ScriptForkAccept(const ForkId& forkId) : forkId_(forkId) {}

  Message toMessage();
  static ScriptForkAccept fromMessage(const Message& message);

  const ForkId forkId_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
