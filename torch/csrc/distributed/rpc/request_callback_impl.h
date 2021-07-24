#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback_no_python.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/jit/python/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

class TORCH_API RequestCallbackImpl : public RequestCallbackNoPython {
 public:
  std::unique_ptr<RpcCommandBase> deserializePythonRpcCommand(
      std::unique_ptr<RpcCommandBase> rpc,
      const MessageType& messageType) const override;

  c10::intrusive_ptr<JitFuture> processPythonCall(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const override;

  c10::intrusive_ptr<JitFuture> processScriptCall(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const override;

  c10::intrusive_ptr<JitFuture> processScriptRemoteCall(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const override;

  c10::intrusive_ptr<JitFuture> processPythonRemoteCall(
      RpcCommandBase& rpc,
      std::vector<c10::Stream> streams) const override;

  c10::intrusive_ptr<JitFuture> processPythonRRefFetchCall(
      RpcCommandBase& rpc) const override;

  void handleRRefDelete(c10::intrusive_ptr<RRef>& rref) const override;

  c10::intrusive_ptr<JitFuture> processRpcWithErrors(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      std::vector<c10::Stream> streams) const override;

  bool cudaAvailable() const override;

  c10::intrusive_ptr<JitFuture> processRRefBackward(
      RpcCommandBase& rpc) const override;

  // Helpers to run user-defined functions, operators and other computations.

  c10::intrusive_ptr<JitFuture> runJitFunction(
      const c10::QualifiedName& name,
      std::vector<at::IValue>& stack,
      std::vector<c10::Stream> streams,
      bool isAsyncExecution) const;

  c10::intrusive_ptr<JitFuture> runPythonFunction(
      const py::object& function,
      std::vector<c10::Stream> streams,
      bool isAsyncExecution) const;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
