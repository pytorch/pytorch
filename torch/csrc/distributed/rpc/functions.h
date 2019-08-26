#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>


namespace torch {
namespace distributed {
namespace rpc {

void processRequestBlocking(
    worker_id_t from, Message&& message, RpcAgent& agent);

} // rpc
} // distributed
} // torch
