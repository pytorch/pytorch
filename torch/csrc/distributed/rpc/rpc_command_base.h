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
  Message toMessage() && {
    JitRRefPickleGuard jitPickleGuard;
    return std::move(*this).toMessageImpl();
  }
  virtual Message toMessageImpl() && = 0;
  virtual ~RpcCommandBase() = 0;
};

// NOLINTNEXTLINE(modernize-use-equals-default)
inline RpcCommandBase::~RpcCommandBase() {}

} // namespace rpc
} // namespace distributed
} // namespace torch
