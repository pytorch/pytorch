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
class TORCH_API ScriptRRefBase {
 public:
  ScriptRRefBase(at::IValue value, MessageType type)
      : value_(std::move(value)), type_(type) {}

  at::IValue value();

  virtual Message toMessage() const;
  static at::IValue fromMessage(const Message& message);

protected:
   const at::IValue value_;
   const MessageType type_;
};

// UserRRef uses this message to fetch the remote RRef value from the owner.
class TORCH_API ScriptRRefFetch final : public ScriptRRefBase {
 public:
  ScriptRRefFetch(at::IValue rrefForkData)
      : ScriptRRefBase(std::move(rrefForkData), MessageType::RREF_FETCH) {}

  static ScriptRRefFetch fromMessage(const Message& message);
};

class TORCH_API PythonRRefFetch final : public ScriptRRefBase {
 public:
  PythonRRefFetch(at::IValue rrefForkData)
      : ScriptRRefBase(std::move(rrefForkData),
                       MessageType::PYTHON_RREF_FETCH) {}

  static PythonRRefFetch fromMessage(const Message& message);
};

// OwnerRRef uses this message to send the RRef value to a remote UserRRef
class TORCH_API ScriptRRefValue final : public ScriptRRefBase {
 public:
  ScriptRRefValue(at::IValue value)
      : ScriptRRefBase(std::move(value), MessageType::RREF_VALUE) {}

  static ScriptRRefValue fromMessage(const Message& message);
};

// UserRRef (regardless it's the creator or not) uses this message to notiify
// OwnerRRef on delete.
class TORCH_API ScriptUserDelete final : public ScriptRRefBase {
 public:
  ScriptUserDelete(at::IValue value)
      : ScriptRRefBase(std::move(value), MessageType::RREF_USER_DELETE) {}

  static ScriptUserDelete fromMessage(const Message& message);
};

// The OwnerRRef uses this message to accept a UserRRef. A UserRRef cannot be
// deleted before receiving this message.
class TORCH_API ScriptUserAccept final : public ScriptRRefBase {
 public:
  ScriptUserAccept(at::IValue value)
      : ScriptRRefBase(std::move(value), MessageType::RREF_USER_ACCEPT) {}

  static ScriptUserAccept fromMessage(const Message& message);
};

// A UserRRef uses this message to notify owner on fork.
class TORCH_API ScriptForkNotify final : public ScriptRRefBase {
 public:
  ScriptForkNotify(at::IValue value, worker_id_t forkDst)
      : ScriptRRefBase(std::move(value), MessageType::RREF_FORK_NOTIFY),
        forkDst_(forkDst) {}

  worker_id_t forkDst() const;

  Message toMessage() const override;
  static ScriptForkNotify fromMessage(const Message& message);

 private:
   const worker_id_t forkDst_;
};

// The OwnerRRef uses this message to a UserRRef that its fork request has been
// accepted. A UserRRef cannot be deleted if it has any pending fork requests.
class TORCH_API ScriptForkAccept final : public ScriptRRefBase {
 public:
   ScriptForkAccept(at::IValue value)
       : ScriptRRefBase(std::move(value), MessageType::RREF_FORK_ACCEPT) {}

   static ScriptForkAccept fromMessage(const Message& message);
};

} // namespace rpc
} // namespace distributed
} // namespace torch
