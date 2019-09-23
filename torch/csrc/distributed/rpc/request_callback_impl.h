#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace rpc {

class TORCH_API RequestCallbackImpl : public RequestCallback {
 public:
  Message processMessage(const Message& request) override;

 private:
  std::unique_ptr<RpcCommandBase> processRpc(
      RpcCommandBase& rpc,
      MessageType messageType);
};

} // namespace rpc
} // namespace distributed
} // namespace torch
