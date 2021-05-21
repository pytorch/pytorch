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
      const c10::intrusive_ptr<JitFuture>& responseFuture) const override;

  void processScriptCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const c10::intrusive_ptr<JitFuture>& responseFuture) const override;

  c10::intrusive_ptr<JitFuture> processScriptRemoteCall(
      ScriptRemoteCall& scriptRemoteCall,
      std::vector<at::IValue>& stack) const override;

  void processPythonRemoteCall(
      RpcCommandBase& rpc,
      const std::function<void(Message)>& markComplete,
      const c10::intrusive_ptr<JitFuture>& responseFuture,
      std::shared_ptr<LazyStreamContext> ctx) const override;

  void processPythonRRefFetchCall(
      RpcCommandBase& rpc,
      const c10::intrusive_ptr<JitFuture>& responseFuture,
      std::shared_ptr<LazyStreamContext> ctx) const override;

  void handleRRefDelete(c10::intrusive_ptr<RRef>& rref) const override;

  c10::intrusive_ptr<JitFuture> processRpcWithErrors(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      std::shared_ptr<LazyStreamContext> ctx) const override;

  bool cudaAvailable() const override;

  void processRRefBackward(
      RpcCommandBase& rpc,
      const c10::intrusive_ptr<JitFuture>& responseFuture) const override;

  // Helpers to run user-defined functions, operators and other computations.

  c10::intrusive_ptr<JitFuture> runJitFunction(
      const c10::QualifiedName& name,
      std::vector<at::IValue>& stack,
      bool isAsyncExecution) const;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
