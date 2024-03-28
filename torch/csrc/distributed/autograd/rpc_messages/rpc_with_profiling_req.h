#pragma once

#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace autograd {

class TORCH_API RpcWithProfilingReq : public rpc::RpcCommandBase {
 public:
  // For sending RPCs, invoked when client is creating this RPC command.
  RpcWithProfilingReq(
      rpc::MessageType messageType,
      c10::intrusive_ptr<rpc::Message> wrappedMessage,
      torch::autograd::profiler::ProfilerConfig&& profilerConfig,
      rpc::ProfilingId profilingKeyId);

  // For receiving an RPC
  // Used in fromMessage.
  RpcWithProfilingReq(
      rpc::MessageType messageType,
      std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
      rpc::MessageType wrappedMessageType,
      std::vector<torch::Tensor> tensors,
      torch::autograd::profiler::ProfilerConfig&& profilerConfig,
      rpc::ProfilingId profilingKeyId);

  // Convert this RPC Command to a Message that can be sent over the wire.
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  static std::unique_ptr<RpcWithProfilingReq> fromMessage(
      const rpc::Message& message);

  // Retrieve the profiling data that is associated with this command.
  torch::autograd::profiler::ProfilerConfig getProfilingConfig() const;
  // Retrieve the globally unique profiling ID corresponding to this command.
  const rpc::ProfilingId& getProfilingId() const;
  // Retrieve the original RPC which this ProfilingRPC wraps.
  RpcCommandBase& wrappedRpc();
  // Destructively move the wrapped RPC.
  std::unique_ptr<RpcCommandBase> moveWrappedRpc() &&;
  // Message type of the wrapped RPC
  rpc::MessageType wrappedMessageType() const;
  void setWrappedRpc(std::unique_ptr<RpcCommandBase> wrappedRpc);

 private:
  // message type
  const rpc::MessageType messageType_;
  // wrapped message
  c10::intrusive_ptr<rpc::Message> wrappedMessage_;
  std::unique_ptr<RpcCommandBase> wrappedRpc_;
  rpc::MessageType wrappedMessageType_;
  std::vector<torch::Tensor> tensors_;
  const torch::autograd::profiler::ProfilerConfig profilerConfig_;
  const rpc::ProfilingId profilingKeyId_;
};
} // namespace autograd
} // namespace distributed
} // namespace torch
