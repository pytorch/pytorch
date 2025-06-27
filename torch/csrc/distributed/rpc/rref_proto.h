#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <vector>

namespace torch::distributed::rpc {

// Temporary solution of RRef operations.
// TODO: Remove all these messages and use rpc + registered functions instead.
class TORCH_API RRefMessageBase : public RpcCommandBase {
 public:
  RRefMessageBase(const RRefId& rrefId, MessageType type)
      : rrefId_(rrefId), type_(type) {}

  const RRefId& rrefId();

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines*)
  const RRefId rrefId_;
  // NOLINTNEXTLINE(cppcoreguidelines*)
  const MessageType type_;
};

class TORCH_API ForkMessageBase : public RRefMessageBase {
 public:
  ForkMessageBase(const RRefId& rrefId, const ForkId& forkId, MessageType type)
      : RRefMessageBase(rrefId, type), forkId_(forkId) {}

  const ForkId& forkId();

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::pair<RRefId, ForkId> fromMessage(
      const Message& message,
      MessageType type);

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines*)
  const ForkId forkId_;
};

// UserRRef uses this message to fetch the remote RRef value from the owner.
class TORCH_API ScriptRRefFetchCall final : public RRefMessageBase {
 public:
  ScriptRRefFetchCall(worker_id_t fromWorkerId, const RRefId& rrefId)
      : RRefMessageBase(rrefId, MessageType::SCRIPT_RREF_FETCH_CALL),
        fromWorkerId_(fromWorkerId) {}

  inline worker_id_t fromWorkerId() const {
    return fromWorkerId_;
  }

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<ScriptRRefFetchCall> fromMessage(
      const Message& message);

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const worker_id_t fromWorkerId_;
};

class TORCH_API PythonRRefFetchCall final : public RRefMessageBase {
 public:
  PythonRRefFetchCall(worker_id_t fromWorkerId, const RRefId& rrefId)
      : RRefMessageBase(rrefId, MessageType::PYTHON_RREF_FETCH_CALL),
        fromWorkerId_(fromWorkerId) {}

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<PythonRRefFetchCall> fromMessage(
      const Message& message);

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const worker_id_t fromWorkerId_;
};

// OwnerRRef uses this message to send the RRef value to a remote UserRRef
class TORCH_API RRefFetchRet : public RpcCommandBase {
 public:
  RRefFetchRet(std::vector<at::IValue> values, MessageType type)
      : values_(std::move(values)), type_(type) {}

  const std::vector<at::IValue>& values();
  c10::intrusive_ptr<Message> toMessageImpl() && override;

 private:
  std::vector<at::IValue> values_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const MessageType type_;
};

class TORCH_API ScriptRRefFetchRet final : public RRefFetchRet {
 public:
  explicit ScriptRRefFetchRet(std::vector<at::IValue> values)
      : RRefFetchRet(std::move(values), MessageType::SCRIPT_RREF_FETCH_RET) {}

  static std::unique_ptr<ScriptRRefFetchRet> fromMessage(
      const Message& message);
};

class TORCH_API PythonRRefFetchRet final : public RRefFetchRet {
 public:
  explicit PythonRRefFetchRet(std::vector<at::IValue> values)
      : RRefFetchRet(std::move(values), MessageType::PYTHON_RREF_FETCH_RET) {}

  static std::unique_ptr<PythonRRefFetchRet> fromMessage(
      const Message& message);
};

// UserRRef (regardless it's the creator or not) uses this message to notify
// OwnerRRef on delete.
class TORCH_API RRefUserDelete final : public ForkMessageBase {
 public:
  RRefUserDelete(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::RREF_USER_DELETE) {}

  static std::unique_ptr<RRefUserDelete> fromMessage(const Message& message);
};

class TORCH_API RemoteRet final : public ForkMessageBase {
 public:
  RemoteRet(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::REMOTE_RET) {}

  static std::unique_ptr<RemoteRet> fromMessage(const Message& message);
};

// A child RRef uses this message to notify its parent that the child has been
// confirmed by the owner.
class TORCH_API RRefChildAccept final : public RpcCommandBase {
 public:
  explicit RRefChildAccept(const ForkId& forkId) : forkId_(forkId) {}

  const ForkId& forkId() const;

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<RRefChildAccept> fromMessage(const Message& message);

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const ForkId forkId_;
};

// A child RRef uses this message to send a fork request to the owner.
class TORCH_API RRefForkRequest final : public ForkMessageBase {
 public:
  RRefForkRequest(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::RREF_FORK_REQUEST) {}

  static std::unique_ptr<RRefForkRequest> fromMessage(const Message& message);
};

class TORCH_API RRefAck final : public RpcCommandBase {
 public:
  RRefAck() = default;

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<RRefAck> fromMessage(const Message& message);
};

} // namespace torch::distributed::rpc
