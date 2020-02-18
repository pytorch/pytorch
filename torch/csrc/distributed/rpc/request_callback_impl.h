#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace rpc {

class TORCH_API RequestCallbackImpl : public RequestCallback {
 public:
  std::shared_ptr<FutureMessage> processMessage(
      Message& request) const override;

 private:
  std::shared_ptr<FutureMessage> processRpc(
      RpcCommandBase& rpc,
      MessageType messageType,
      int64_t messageId) const;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
