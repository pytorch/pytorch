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
    worker_id_t from, Message&& message, RpcAgent& agent);

} // rpc
} // distributed
} // torch
