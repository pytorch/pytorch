#pragma once

#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace autograd {
// TODO: Both this class and RpcWithAutograd wrap other RPC message types with
// some information. We can dedupe these into a generic WrappedRpc class,
// which these classes can inherit from.
class TORCH_API RpcWithProfilingReq : public rpc::RpcCommandBase {
 public:
  // For sending RPCs, invoked when client is creating this RPC command.
  RpcWithProfilingReq(
      rpc::worker_id_t fromWorkerId,
      rpc::MessageType messageType,
      rpc::Message&& wrappedMessage,
      const torch::autograd::profiler::ProfilerConfig& profilerConfig,
      const std::string profilingKey);

  // For receiving an RPC
  // Used in fromMessage.
  RpcWithProfilingReq(
      rpc::worker_id_t fromWorkerId,
      rpc::MessageType messageType,
      std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
      rpc::MessageType wrappedMessageType,
      std::vector<torch::Tensor> tensors,
      torch::autograd::profiler::ProfilerConfig& profilerConfig,
      const std::string clientProfilingKey);

  // Convert this RPC Command to a Message that can be sent over the wire.
  rpc::Message toMessageImpl() && override;
  static std::unique_ptr<RpcWithProfilingReq> fromMessage(
      const rpc::Message& message);

  // Retrieve the profiling data that is associated with this command.
  torch::autograd::profiler::ProfilerConfig getProfilingConfig() const;
  // Retrieve the profiling key that is associated with this command
  const std::string getProfilingKey() const;
  // Retrieve the original RPC which this ProfilingRPC wraps.
  RpcCommandBase& wrappedRpc();
  // Destructively move the wrapped RPC.
  std::unique_ptr<RpcCommandBase> moveWrappedRpc() &&;
  // Message type of the wrapped RPC
  rpc::MessageType wrappedMessageType() const;
  // retrieve the WID from which the rpc came from
  rpc::worker_id_t fromWorkerId() const;
  void setWrappedRpc(std::unique_ptr<RpcCommandBase> wrappedRpc);

 private:
  // the worker id
  rpc::worker_id_t fromWorkerId_;
  // message type
  rpc::MessageType messageType_;
  // wrapped message
  rpc::Message wrappedMessage_;
  std::unique_ptr<RpcCommandBase> wrappedRpc_;
  rpc::MessageType wrappedMessageType_;
  std::vector<torch::Tensor> tensors_;
  torch::autograd::profiler::ProfilerConfig profilerConfig_;
  std::string profilingKey_;
};
} // namespace autograd
} // namespace distributed
} // namespace torch
