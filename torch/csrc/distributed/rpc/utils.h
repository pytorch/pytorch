#pragma once

#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace rpc {

// Given an RPC message received as a request over the wire, deserialize it into
// the appropriate 'RpcCommandBase' type.
TORCH_API std::unique_ptr<RpcCommandBase> deserializeRequest(
    const Message& request);

// Given an RPC message received as a response over the wire, deserialize it
// into the appropriate 'RpcCommandBase' type.
TORCH_API std::unique_ptr<RpcCommandBase> deserializeResponse(
    const Message& response);

} // namespace rpc
} // namespace distributed
} // namespace torch
