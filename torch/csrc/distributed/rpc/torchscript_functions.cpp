#include <torch/csrc/distributed/rpc/torchscript_functions.h>
#include <ATen/ThreadLocalState.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

c10::intrusive_ptr<c10::ivalue::Future> rpcTorchscript(
    const std::string& dstWorkerName,
    const c10::QualifiedName& qualifiedName,
    const c10::FunctionSchema& functionSchema,
    std::vector<c10::IValue>& stack,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution) {
  if (torch::autograd::profiler::profilerEnabled() &&
      !torch::distributed::rpc::RemoteProfilerManager::getInstance()
           .isCurrentKeySet()) {
    auto rpcAsyncJitKey = fmt::format(
        "rpc_async_jit#({})->({})",
        RpcAgent::getCurrentRpcAgent()->getWorkerInfo().name_,
        dstWorkerName);
    auto& remoteProfilerManager =
        torch::distributed::rpc::RemoteProfilerManager::getInstance();
    remoteProfilerManager.setCurrentKey(rpcAsyncJitKey);
  }
  auto scriptCall = std::make_unique<ScriptCall>(
      qualifiedName, std::move(stack), isAsyncExecution);
  auto rpcAgentPtr = RpcAgent::getCurrentRpcAgent();
  auto futMessage = autograd::sendMessageWithAutograd(
      *rpcAgentPtr,
      rpcAgentPtr->getWorkerInfo(dstWorkerName),
      std::move(*scriptCall).toMessage(),
      true /*forceGradRecording*/,
      rpcTimeoutSeconds);

  // Get function return type to construct c10::ivalue::Future.
  auto returns = functionSchema.returns();
  // Script call only allows single IValue returned.
  TORCH_INTERNAL_ASSERT(
      returns.size() == 1,
      "Return value of an annotated torchScript function should be a single "
      "IValue.",
      returns.size());
  auto returnType = returns.at(0).type();

  // Create a JIT future and pass it to futMessage's callback to set state
  // of the JIT future.
  auto futPtr = c10::make_intrusive<c10::ivalue::Future>(returnType);
  // Save and pass thread local state into the callback
  at::ThreadLocalState tls_state;
  futMessage->addCallback([futPtr, tls_state = std::move(tls_state)](
                              const FutureMessage& futMessage) {
    at::ThreadLocalStateGuard g(tls_state);
    if (futMessage.hasError()) {
      c10::ivalue::Future::FutureError jitFutErr(futMessage.error()->what());
      futPtr->setError(std::move(jitFutErr));
    } else {
      futPtr->markCompleted(deserializeRespToIValue(futMessage.constValue()));
    }
  });
  return futPtr;
}

c10::intrusive_ptr<RRef> remoteTorchscript(
    const std::string& dstWorkerName,
    const c10::QualifiedName& qualifiedName,
    const c10::FunctionSchema& functionSchema,
    std::vector<c10::IValue>& stack,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution) {
  auto rpcAgentPtr = RpcAgent::getCurrentRpcAgent();
  auto dstWorkerInfo = rpcAgentPtr->getWorkerInfo(dstWorkerName);
  auto& ctx = RRefContext::getInstance();

  // Get function return type to construct UserRRef.
  auto returns = functionSchema.returns();
  // Script call only allows single IValue returned.
  TORCH_INTERNAL_ASSERT(
      returns.size() == 1,
      "Return value of an annotated torchScript function should be a single "
      "IValue.",
      returns.size());
  auto returnType = returns.at(0).type();

  at::ThreadLocalState tls_state;
  if (ctx.getWorkerId() != dstWorkerInfo.id_) {
    auto userRRefPtr = ctx.createUserRRef(dstWorkerInfo.id_, returnType);

    auto scriptRemoteCall = std::make_unique<ScriptRemoteCall>(
        qualifiedName,
        std::move(stack),
        userRRefPtr->rrefId(),
        userRRefPtr->forkId(),
        isAsyncExecution);

    auto fm = torch::distributed::autograd::sendMessageWithAutograd(
        *rpcAgentPtr,
        dstWorkerInfo,
        std::move(*scriptRemoteCall).toMessage(),
        true /*forceGradRecording*/,
        rpcTimeoutSeconds /* timeout */);

    userRRefPtr->registerOwnerCreationFuture(fm);

    ctx.addPendingUser(userRRefPtr->forkId(), userRRefPtr);
    fm->addCallback(
        [forkId{userRRefPtr->forkId()},
         tls_state = std::move(tls_state)](const FutureMessage& fm) {
          at::ThreadLocalStateGuard g(tls_state);
          callback::confirmPendingUser(fm, forkId);
        });

    return userRRefPtr;
  } else {
    auto ownerRRefPtr = ctx.createOwnerRRef(returnType);
    // prevent this owner RRef from being deleted due to other forks
    ctx.addSelfAsFork(ownerRRefPtr);

    auto scriptRemoteCall = std::make_unique<ScriptRemoteCall>(
        qualifiedName,
        std::move(stack),
        ownerRRefPtr->rrefId(),
        ownerRRefPtr->rrefId(),
        isAsyncExecution);

    auto fm = torch::distributed::autograd::sendMessageWithAutograd(
        *rpcAgentPtr,
        dstWorkerInfo,
        std::move(*scriptRemoteCall).toMessage(),
        true /*forceGradRecording*/,
        rpcTimeoutSeconds /* timeout */);

    ownerRRefPtr->registerOwnerCreationFuture(fm);
    fm->addCallback(
        [tls_state = std::move(tls_state),
         ownerRRefId = ownerRRefPtr->rrefId()](const FutureMessage& fm) {
          at::ThreadLocalStateGuard g(tls_state);
          callback::finishCreatingOwnerRRef(fm, ownerRRefId);
        });
    return ownerRRefPtr;
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
