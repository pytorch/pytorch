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
  RRefMessageBase(const RRefId& rrefId, MessageType type)
      : rrefId_(rrefId), type_(type) {}

  const RRefId& rrefId();

  virtual Message toMessage() const;
  static at::IValue fromMessage(const Message& message, MessageType type);

 protected:
  const RRefId rrefId_;
  const MessageType type_;
};

class TORCH_API ForkMessageBase : public RRefMessageBase {
 public:
  ForkMessageBase(const RRefId& rrefId, const ForkId& forkId, MessageType type)
      : RRefMessageBase(rrefId, type), forkId_(forkId) {}

  const ForkId& forkId();

  virtual Message toMessage() const;
  static std::pair<RRefId, ForkId> fromMessage(
      const Message& message,
      MessageType type);

 protected:
  const ForkId forkId_;
};

// UserRRef uses this message to fetch the remote RRef value from the owner.
class TORCH_API ScriptRRefFetchCall final : public RRefMessageBase {
 public:
  ScriptRRefFetchCall(const RRefId& rrefId)
      : RRefMessageBase(rrefId, MessageType::SCRIPT_RREF_FETCH_CALL) {}

  static ScriptRRefFetchCall fromMessage(const Message& message);
};

class TORCH_API PythonRRefFetchCall final : public RRefMessageBase {
 public:
  PythonRRefFetchCall(const RRefId& rrefId)
      : RRefMessageBase(rrefId, MessageType::PYTHON_RREF_FETCH_CALL) {}

  static PythonRRefFetchCall fromMessage(const Message& message);
};

// OwnerRRef uses this message to send the RRef value to a remote UserRRef
class TORCH_API RRefFetchRet final {
 public:
  RRefFetchRet(at::IValue value) : value_(std::move(value)) {}

  const at::IValue& value();

  Message toMessage() const;
  static RRefFetchRet fromMessage(const Message& message);

 private:
  at::IValue value_;
};

// UserRRef (regardless it's the creator or not) uses this message to notiify
// OwnerRRef on delete.
class TORCH_API RRefUserDelete final : public ForkMessageBase {
 public:
  RRefUserDelete(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::RREF_USER_DELETE) {}

  static RRefUserDelete fromMessage(const Message& message);
};

// The OwnerRRef uses this message to accept a UserRRef. A UserRRef cannot be
// deleted before receiving this message.
class TORCH_API RRefUserAccept final : public ForkMessageBase {
 public:
  RRefUserAccept(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::RREF_USER_ACCEPT) {}

  static RRefUserAccept fromMessage(const Message& message);
};

class TORCH_API RemoteRet final : public ForkMessageBase {
 public:
  RemoteRet(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::REMOTE_RET) {}

  static RemoteRet fromMessage(const Message& message);
};

// The OwnerRRef uses this message to a UserRRef that its fork request has been
// accepted. A UserRRef cannot be deleted if it has any pending fork requests.

class TORCH_API RRefChildAccept final {
 public:
  RRefChildAccept(const ForkId& forkId) : forkId_(forkId) {}

  const ForkId& forkId() const;

  Message toMessage();
  static RRefChildAccept fromMessage(const Message& message);

 private:
  const ForkId forkId_;
};

class TORCH_API RRefForkRequest final : public ForkMessageBase {
 public:
  RRefForkRequest(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::RREF_FORK_REQUEST) {}

  static RRefForkRequest fromMessage(const Message& message);
};

} // namespace rpc
} // namespace distributed
} // namespace torch
