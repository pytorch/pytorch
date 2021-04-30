#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>

namespace torch {
namespace distributed {
namespace rpc {

// RequestCallback implementation with no Python dependencies.
class TORCH_API RequestCallbackNoPython : public RequestCallback {
 public:
  std::shared_ptr<JitFuture> processMessage(Message& request) const override;

 protected:
  virtual std::unique_ptr<RpcCommandBase> deserializePythonRpcCommand(
      std::unique_ptr<RpcCommandBase> rpc,
      const MessageType& messageType) const;

  virtual void processScriptCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const;

  bool processScriptCallOp(
      ScriptCall& scriptCall,
      const std::function<void(Message)>& markComplete,
      std::vector<at::IValue>& stack) const;

  virtual void processPythonCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const;

  virtual TypePtr getScriptRemoteCallType(
      ScriptRemoteCall& scriptRemoteCall) const;

  virtual void processScriptRemoteCall(
      ScriptRemoteCall& scriptRemoteCall,
      const std::function<void(void)>& postProcessing,
      std::vector<at::IValue>& stack,
      const c10::intrusive_ptr<OwnerRRef>& ownerRRef) const;

  void processBaseScriptRemoteCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const;

  bool processScriptRemoteCallOp(
      ScriptRemoteCall& scriptRemoteCall,
      const std::function<void(void)>& postProcessing,
      std::vector<at::IValue>& stack,
      const c10::intrusive_ptr<OwnerRRef>& ownerRRef) const;

  virtual void processPythonRemoteCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const;

  void processScriptRRefFetchCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const;

  virtual void processPythonRRefFetchCall(
      RpcCommandBase& rpc,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const;

  void processRRefUserDelete(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete) const;

  void processRRefChildAccept(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete) const;

  void processRRefForkRequest(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete) const;

  void processForwardAutogradReq(
      RpcCommandBase& rpc,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const;

  void processBackwardAutogradReq(
      RpcCommandBase& rpc,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const;

  void processCleanupAutogradContextReq(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete) const;

  void processRunWithProfilingReq(
      RpcCommandBase& rpc,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const;

  virtual void handleRRefDelete(c10::intrusive_ptr<RRef>& rref) const;

  void processRpc(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const;

  virtual void processRpcWithErrors(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const;

  IValue handleError(
      const std::exception& e,
      const MessageType messageType,
      int64_t messageId) const;

  virtual bool cudaAvailable() const;

  virtual void processRRefBackward(
      RpcCommandBase& rpc,
      const int64_t messageId,
      const std::shared_ptr<JitFuture>& responseFuture) const;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
