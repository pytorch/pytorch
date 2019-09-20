#pragma once

#include <torch/csrc/distributed/rpc/message.h>

namespace torch {
namespace distributed {
namespace rpc {

// Base class for all RPC request and responses.
class RpcCommandBase {
 public:
  // Need to override this to serialize the RPC.
  virtual Message toMessage() = 0;
  virtual ~RpcCommandBase() = 0;
};

inline RpcCommandBase::~RpcCommandBase() {}

} // namespace rpc
} // namespace distributed
} // namespace torch
