#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace rpc {

// Base class for all RPC request and responses.
class RpcCommandBase {
 public:
  // Need to override this to serialize the RPC. This should destructively
  // create a message for the RPC (Hence the &&).
  c10::intrusive_ptr<OutgoingMessage> toMessage(const std::string& meta) && {
    JitRRefPickleGuard jitPickleGuard;
    return std::move(*this).toMessageImpl(meta);
  }
  virtual c10::intrusive_ptr<OutgoingMessage> toMessageImpl(const std::string& meta) && = 0;
  virtual ~RpcCommandBase() = 0;
};

inline RpcCommandBase::~RpcCommandBase() = default;

} // namespace rpc
} // namespace distributed
} // namespace torch
