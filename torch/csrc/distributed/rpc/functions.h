#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>


namespace torch {
namespace distributed {
namespace rpc {

void processRequestBlocking(
    const WorkerId& from, Message&& message, RpcAgent& agent);

} // rpc
} // distributed
} // torch
