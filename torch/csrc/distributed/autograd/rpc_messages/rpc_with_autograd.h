#pragma once

#include <torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace autograd {

// Represents an RPC that includes autograd information. This class basically
// wraps another `RpcCommandBase` object which represents the actual RPC and has
// additional autograd information associated with that RPC.
class TORCH_API RpcWithAutograd final : public rpc::RpcCommandBase {
 public:
  // Used when we are sending an RPC over the wire.
  RpcWithAutograd(
      rpc::worker_id_t fromWorkerId,
      rpc::MessageType messageType,
      const AutogradMetadata& autogradMetadata,
      rpc::Message&& wrappedMessage,
      std::unordered_map<c10::DeviceIndex, c10::DeviceIndex> deviceMap = {});

  // Used when receiving an RPC over the wire.
  RpcWithAutograd(
      rpc::worker_id_t fromWorkerId,
      rpc::MessageType messageType,
      const AutogradMetadata& autogradMetadata,
      std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
      rpc::MessageType wrappedMessageType,
      std::vector<torch::Tensor> tensors,
      std::unordered_map<c10::DeviceIndex, c10::DeviceIndex> deviceMap = {});

  rpc::Message toMessageImpl() && override;

  static std::unique_ptr<RpcWithAutograd> fromMessage(
      const rpc::Message& message);

  // Retrieves tensors as part of this RPC, which need to be considered for
  // autograd computations.
  std::vector<torch::Tensor>& tensors();

  const AutogradMetadata& autogradMetadata() const;

  RpcCommandBase& wrappedRpc();

  void setWrappedRpc(std::unique_ptr<RpcCommandBase> wrappedRpc);

  std::unique_ptr<RpcCommandBase> moveWrappedRpc() &&;

  // Message type of the wrapped RPC.
  rpc::MessageType wrappedMessageType() const;

  // Retrieve the worker id from which the RPC originated.
  rpc::worker_id_t fromWorkerId() const;

  // Retrieve the device map.
  const std::unordered_map<c10::DeviceIndex, c10::DeviceIndex>& deviceMap();

 private:
  // WorkerId from which this RPC originated. This is necessary for knowing
  // which worker we need to contact during the backward pass.
  rpc::worker_id_t fromWorkerId_;

  // Message type for this call.
  rpc::MessageType messageType_;

  AutogradMetadata autogradMetadata_;

  // Since wrappedMessage_ is destructively constructed from wrappedRpc_,
  // they are valid exclusively. They are used for different purpose.
  // wrappedRpc_ is used while constructing receive rpcWithAutograd;
  // wrappedMessage_ is used while constructing send rpcWithAutograd;

  // When receive rpcWithAutograd is constructed fromMessage, it is valid;
  // When send rpcWithAutograd is constructed before toMessage, it is nullptr;
  std::unique_ptr<RpcCommandBase> wrappedRpc_;

  // Serialized message representing wrappedRpc_. Used mostly as a cache to
  // avoid serializing the request twice.
  // When receive rpcWithAutograd is constructed fromMessage, it is nullptr;
  // When send rpcWithAutograd is constructed before toMessage, it is valid;
  rpc::Message wrappedMessage_;

  // message type of the wrappedMessage, this is stored separately since
  // wrappedMessage_ is not always guaranteed to be populated.
  rpc::MessageType wrappedMessageType_;

  // Tensors part of the wrappedRpc that need to be considered for autograd.
  std::vector<torch::Tensor> tensors_;

  // Device mapping for tensors that are sent across an RPC to another node.
  std::unordered_map<c10::DeviceIndex, c10::DeviceIndex> deviceMap_;
};

} // namespace autograd
} // namespace distributed
} // namespace torch
