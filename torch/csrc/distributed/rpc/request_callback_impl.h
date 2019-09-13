#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback.h>
#include <torch/csrc/distributed/rpc/rpc_base.h>

namespace torch {
namespace distributed {
namespace rpc {

class RequestCallbackImpl : public RequestCallback {
 public:
  Message processMessage(Message& request) override;

 private:
  std::unique_ptr<RpcBase> processRpc(RpcBase* rpc, MessageType messageType);
};

} // namespace rpc
} // namespace distributed
} // namespace torch
