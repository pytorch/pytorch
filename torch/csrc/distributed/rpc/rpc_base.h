#pragma once

#include <torch/csrc/distributed/rpc/message.h>

namespace torch {
namespace distributed {
namespace rpc {

// Base class for all RPC request and responses.
class RpcBase {
 public:
  // Need to override this to serialize the RPC.
  virtual Message toMessage() = 0;
  virtual ~RpcBase() = 0;
};

inline RpcBase::~RpcBase() {}

} // namespace rpc
} // namespace distributed
} // namespace torch
