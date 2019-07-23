#pragma once

#include <torch/csrc/distributed/rpc/BuiltinOp.h>
#include <torch/csrc/distributed/rpc/BuiltinRet.h>
#include <torch/csrc/distributed/rpc/FutureMessage.h>
#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/RpcAgent.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>


namespace torch {
namespace distributed {
namespace rpc {

void processRequestBlocking(std::string from, Message message, RpcAgent& agent);

} // rpc
} // distributed
} // torch
