#pragma once

#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_ret.h>

namespace torch {
namespace distributed {
namespace rpc {

Message processRequestBlocking(Message&& message);

Message createException(const Message& request, const std::exception& e);

} // namespace rpc
} // namespace distributed
} // namespace torch
