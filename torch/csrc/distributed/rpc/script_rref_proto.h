#pragma once

#include <torch/csrc/distributed/rpc/message.h>
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

  Message toMessage() const;
  static at::IValue fromMessage(const Message& message);

 private:
   const at::IValue value_;
   const MessageType type_;
};

class TORCH_API ScriptRRefFetch final : public ScriptRRefBase {
 public:
  ScriptRRefFetch(at::IValue rrefForkData)
      : ScriptRRefBase(std::move(rrefForkData), MessageType::RREF_FETCH) {}

  static ScriptRRefFetch fromMessage(const Message& message);
};

class TORCH_API ScriptRRefValue final : public ScriptRRefBase {
 public:
  ScriptRRefValue(at::IValue value)
      : ScriptRRefBase(std::move(value), MessageType::RREF_VALUE) {}

  static ScriptRRefValue fromMessage(const Message& message);
};

class TORCH_API ScriptRRefAdd final : public ScriptRRefBase {
 public:
  ScriptRRefAdd(at::IValue value)
      : ScriptRRefBase(std::move(value), MessageType::RREF_ADD_FORK) {}

  static ScriptRRefAdd fromMessage(const Message& message);
};

class TORCH_API ScriptRRefDel final : public ScriptRRefBase {
 public:
  ScriptRRefDel(at::IValue value)
      : ScriptRRefBase(std::move(value), MessageType::RREF_DEL_FORK) {}

  static ScriptRRefDel fromMessage(const Message& message);
};

} // namespace rpc
} // namespace distributed
} // namespace torch
