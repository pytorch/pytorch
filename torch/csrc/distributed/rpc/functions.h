#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace rpc {

Message processRequestBlocking(Message&& request);

Message createException(const Message& request, const std::exception& e);

} // namespace rpc
} // namespace distributed
} // namespace torch
