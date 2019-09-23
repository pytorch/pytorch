#pragma once

#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace rpc {

struct TORCH_API AutogradMetadata {
  AutogradMetadata(int64_t autogradContextId, int64_t autogradMessageId);

  // autogradContextId_ is a globally unique integer that identifies a
  // particular distributed autograd pass.
  int64_t autogradContextId;
  // autogradMessageId_ is a globally unique integer that identifies a pair
  // of send/recv autograd functions.
  int64_t autogradMessageId;
};

// Represents an RPC that includes autograd information. This class basically
// wraps another `RpcCommandBase` object which represents the actual RPC and has
// additional autograd information associated with that RPC.
class TORCH_API RpcWithAutograd final : public RpcCommandBase {
 public:
  RpcWithAutograd(
      MessageType messageType,
      const AutogradMetadata& autogradMetadata,
      std::unique_ptr<RpcCommandBase> wrappedRpc);

  // This variant is used when we already have a serialized message for
  // wrappedRpc.
  RpcWithAutograd(
      MessageType messageType,
      const AutogradMetadata& autogradMetadata,
      std::unique_ptr<RpcCommandBase> wrappedRpc,
      MessageType wrappedMessageType,
      std::vector<torch::Tensor> tensors);

  // Destructively creates a message to avoid copies.
  Message toMessage() override;

  static std::unique_ptr<RpcWithAutograd> fromMessage(const Message& message);

  // Retrieves tensors as part of this RPC, which need to be considered for
  // autograd computations.
  std::vector<torch::Tensor>& tensors();

  const AutogradMetadata& autogradMetadata() const;

  // Destructively retrieves the wrapped rpc.
  std::unique_ptr<RpcCommandBase> moveWrappedRpc() &&;

  // Message type of the wrapped RPC.
  MessageType wrappedMessageType() const;

 private:
  // Message type for this call.
  MessageType messageType_;

  AutogradMetadata autogradMetadata_;
  std::unique_ptr<RpcCommandBase> wrappedRpc_;

  // Serialized message representing wrappedRpc_. Used mostly as a cache to
  // avoid serializing the request twice.
  Message wrappedMessage_;

  // message type of the wrappedMessage, this is stored separately since
  // wrappedMessage_ is not always guaranteed to be populated.
  MessageType wrappedMessageType_;

  // Tensors part of the wrappedRpc that need to be considered for autograd.
  std::vector<torch::Tensor> tensors_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
