#pragma once

#include <torch/csrc/distributed/rpc/rpc_base.h>

namespace torch {
namespace distributed {
namespace rpc {

// Given an RPC message received as a request over the wire, deserialize it into
// the appropriate 'RpcBase' type.
std::unique_ptr<RpcBase> deserializeRequest(const Message& request);

// Given an RPC message received as a response over the wire, deserialize it
// into the appropriate 'RpcBase' type.
std::unique_ptr<RpcBase> deserializeResponse(const Message& response);

} // namespace rpc
} // namespace distributed
} // namespace torch
