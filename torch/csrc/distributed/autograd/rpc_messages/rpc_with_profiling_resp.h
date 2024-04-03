#pragma once

#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace autograd {
class TORCH_API RpcWithProfilingResp : public rpc::RpcCommandBase {
 public:
  // For sending RPCs over the wire
  RpcWithProfilingResp(
      rpc::MessageType messageType,
      c10::intrusive_ptr<rpc::Message> wrappedMessage,
      std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents,
      rpc::ProfilingId profilingId);

  // For receiving RPCs. Used in from message when converting a message received
  // over the wire.
  RpcWithProfilingResp(
      rpc::MessageType messageType,
      std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
      rpc::MessageType wrappedMessageType,
      std::vector<torch::Tensor> tensors,
      std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents,
      rpc::ProfilingId profilingId);
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  static std::unique_ptr<RpcWithProfilingResp> fromMessage(
      const rpc::Message& message);
  // Retrieve remote Events
  std::vector<torch::autograd::profiler::LegacyEvent> getProfiledEvents() const;
  // Retrieve the globally unique profiling ID corresponding to this command.
  const rpc::ProfilingId& getProfilingId() const;
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
  c10::intrusive_ptr<rpc::Message> wrappedMessage_;
  std::unique_ptr<RpcCommandBase> wrappedRpc_;
  rpc::MessageType wrappedMessageType_;
  std::vector<torch::Tensor> tensors_;
  const std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents_;
  const rpc::ProfilingId profilingId_;
};
} // namespace autograd
} // namespace distributed
} // namespace torch
