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
  void processRpc(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      const int64_t messageId,
      const std::shared_ptr<FutureMessage>& retFutureMessagge) const;

  Message handleError(
      const std::exception& e,
      const MessageType messageType,
      int64_t messageId) const;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
