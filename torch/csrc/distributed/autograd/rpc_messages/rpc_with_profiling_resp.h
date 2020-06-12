#pragma once

#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace autograd {
class TORCH_API RpcWithProfilingResp : public rpc::RpcCommandBase {
 public:
  // For sending RPCs over the wire
  RpcWithProfilingResp(
      rpc::MessageType messageType,
      rpc::Message&& wrappedMessage,
      std::vector<torch::autograd::profiler::Event> profiledEvents);

  // For receving RPCs. Used in from message when converting a message received
  // over the wire.
  RpcWithProfilingResp(
      rpc::MessageType messageType,
      std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
      rpc::MessageType wrappedMessageType,
      std::vector<torch::Tensor> tensors,
      std::vector<torch::autograd::profiler::Event> profiledEvents);
  rpc::Message toMessageImpl() && override;
  static std::unique_ptr<RpcWithProfilingResp> fromMessage(
      const rpc::Message& message);
  // Retrieve remote Events
  std::vector<torch::autograd::profiler::Event> getProfiledEvents() const;
  // Retrieve the original RPC which this ProfilingRPC wraps.
  RpcCommandBase& wrappedRpc();
  // Destructively move the wrapped RPC.
  std::unique_ptr<RpcCommandBase> moveWrappedRpc() &&;
  // Message type of the wrapped RPC
  rpc::MessageType wrappedMessageType() const;
  // Set the wrapped RPC for this RPC.
  void setWrappedRpc(std::unique_ptr<RpcCommandBase> wrappedRpc);

 private:
  // message type
  const rpc::MessageType messageType_;
  // wrapped message
  rpc::Message wrappedMessage_;
  std::unique_ptr<RpcCommandBase> wrappedRpc_;
  rpc::MessageType wrappedMessageType_;
  std::vector<torch::Tensor> tensors_;
  const std::vector<torch::autograd::profiler::Event> profiledEvents_;
};
} // namespace autograd
} // namespace distributed
} // namespace torch
