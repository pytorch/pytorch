#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback_no_python.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace rpc {

class TORCH_API RequestCallbackImpl : public RequestCallbackNoPython {
 public:
  std::unique_ptr<RpcCommandBase> deserializePythonRpcCommand(
      std::unique_ptr<RpcCommandBase> rpc,
      const MessageType& messageType) const override;

  void processPythonCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const override;

  void processScriptCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const override;

  TypePtr getScriptRemoteCallType(
      ScriptRemoteCall& scriptRemoteCall) const override;

  void processScriptRemoteCall(
      ScriptRemoteCall& scriptRemoteCall,
      const std::function<void(void)>& postProcessing,
      std::vector<at::IValue>& stack,
      const c10::intrusive_ptr<OwnerRRef>& ownerRRef) const override;

  void processPythonRemoteCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const override;

  void processPythonRRefFetchCall(
      RpcCommandBase& rpc,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const override;

  void handleRRefDelete(c10::intrusive_ptr<RRef>& rref) const override;

  void processRpcWithErrors(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const override;

  bool cudaAvailable() const override;

  void processRRefBackward(
      RpcCommandBase& rpc,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const override;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
