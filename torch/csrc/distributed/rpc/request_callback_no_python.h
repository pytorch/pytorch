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
  c10::intrusive_ptr<JitFuture> processMessage(
      Message& request,
      std::vector<c10::Stream> streams) const override;

 protected:
  virtual std::unique_ptr<RpcCommandBase> deserializePythonRpcCommand(
      std::unique_ptr<RpcCommandBase> rpc,
      const MessageType& messageType) const;

  virtual c10::intrusive_ptr<JitFuture> processScriptCall(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const;

  virtual c10::intrusive_ptr<JitFuture> processPythonCall(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const;

  c10::intrusive_ptr<JitFuture> assignOwnerRRef(
      const RRefId& rrefId,
      const RRefId& forkId,
      c10::intrusive_ptr<JitFuture> valueFuture) const;

  virtual c10::intrusive_ptr<JitFuture> processScriptRemoteCall(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const;

  virtual c10::intrusive_ptr<JitFuture> processPythonRemoteCall(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const;

  c10::intrusive_ptr<JitFuture> retrieveOwnerRRef(const RRefId& rrefId) const;

  c10::intrusive_ptr<JitFuture> processScriptRRefFetchCall(
      RpcCommandBase& rpc) const;

  virtual c10::intrusive_ptr<JitFuture> processPythonRRefFetchCall(
      RpcCommandBase& rpc) const;

  c10::intrusive_ptr<JitFuture> processRRefUserDelete(
      RpcCommandBase& rpc) const;

  c10::intrusive_ptr<JitFuture> processRRefChildAccept(
      RpcCommandBase& rpc) const;

  c10::intrusive_ptr<JitFuture> processRRefForkRequest(
      RpcCommandBase& rpc) const;

  c10::intrusive_ptr<JitFuture> processForwardAutogradReq(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const;

  c10::intrusive_ptr<JitFuture> processBackwardAutogradReq(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const;

  c10::intrusive_ptr<JitFuture> processCleanupAutogradContextReq(
      RpcCommandBase& rpc) const;

  c10::intrusive_ptr<JitFuture> processRunWithProfilingReq(
      RpcCommandBase& rpc) const;

  virtual void handleRRefDelete(c10::intrusive_ptr<RRef>& rref) const;

  c10::intrusive_ptr<JitFuture> processRpc(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      std::vector<c10::Stream> streams) const;

  virtual c10::intrusive_ptr<JitFuture> processRpcWithErrors(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      std::vector<c10::Stream> streams) const;

  c10::intrusive_ptr<Message> handleError(
      const std::exception& e,
      const MessageType messageType,
      int64_t messageId) const;

  virtual bool cudaAvailable() const;

  virtual c10::intrusive_ptr<JitFuture> processRRefBackward(
      RpcCommandBase& rpc) const;

  // Helpers to run user-defined functions, operators and other computations.

  c10::intrusive_ptr<JitFuture> runJitOperator(
      const jit::Operator& op,
      std::vector<at::IValue>& stack,
      std::vector<c10::Stream> streams) const;

  // Helpers to convert various kinds of objects into already-completed futures.

  c10::intrusive_ptr<JitFuture> asFuture(IValue value, TypePtr type) const;

  c10::intrusive_ptr<JitFuture> asFuture(
      c10::intrusive_ptr<Message> message) const;

  c10::intrusive_ptr<JitFuture> asFuture(std::exception_ptr err) const;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
