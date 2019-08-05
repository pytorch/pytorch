#pragma once

#include <torch/csrc/distributed/rpc/FutureMessage.h>
#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/RpcAgent.h>
#include <torch/csrc/distributed/rpc/ScriptCall.h>
#include <torch/csrc/distributed/rpc/ScriptRet.h>

namespace torch {
namespace distributed {
namespace rpc {

void processRequestBlocking(
    const std::string& from, Message&& message, RpcAgent& agent);

} // rpc
} // distributed
} // torch
