#include <torch/csrc/distributed/rpc/request_callback_no_python.h>

#include <c10/core/StreamGuard.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/distributed/rpc/utils.h>

#include <utility>

namespace torch::distributed::rpc {

using namespace torch::distributed::autograd;
using namespace torch::autograd::profiler;

// When request message has autograd info, processMessage() will set up valid
// current context id properly. This struct is used to clean up current context
// id after processMessage() is done.
struct DistAutogradContextGuard {
  explicit DistAutogradContextGuard(int64_t ctxId) {
    auto& container = DistAutogradContainer::getInstance();
    prevCtxId_ = container.currentContextId();
    container.forceCurrentContextId(ctxId);
  }
  DistAutogradContextGuard(const DistAutogradContextGuard&) = delete;
  DistAutogradContextGuard(DistAutogradContextGuard&&) = delete;
  DistAutogradContextGuard& operator=(const DistAutogradContextGuard&) = delete;
  DistAutogradContextGuard& operator=(DistAutogradContextGuard&&) = delete;
  ~DistAutogradContextGuard() {
    auto& container = DistAutogradContainer::getInstance();
    container.forceCurrentContextId(prevCtxId_);
  }

  int64_t prevCtxId_;
};

std::unique_ptr<RpcCommandBase> RequestCallbackNoPython::
    deserializePythonRpcCommand(
        std::unique_ptr<RpcCommandBase> rpc,
        const MessageType& messageType) const {
  TORCH_CHECK(
      messageType != MessageType::PYTHON_CALL &&
          messageType != MessageType::PYTHON_REMOTE_CALL,
      "Python calls are not supported!");
  return rpc;
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processMessage(
    Message& request,
    std::vector<c10::Stream> streams) const {
  // We need two futures here because it could pause twice when processing a
  // RPC message:
  //  1) waiting for all RRefs in the arguments to become confirmed;
  //  2) waiting for processRpc to finish.
  auto& rrefContext = RRefContext::getInstance();
  try {
    rrefContext.recordThreadLocalPendingRRefs();
    // Deserialize PythonUDF here to trigger RRef unpickling
    std::unique_ptr<RpcCommandBase> rpc = deserializePythonRpcCommand(
        deserializeRequest(request), request.type());
    auto rrefsReadyFuture = rrefContext.waitForThreadLocalPendingRRefs();

    auto retFuture = rrefsReadyFuture->thenAsync(
        [this,
         // std::function must be copyable, hence has to cast the unique_ptr to
         // a shared_ptr here.
         rpc = std::shared_ptr<RpcCommandBase>(std::move(rpc)),
         messageType = request.type(),
         streams = std::move(streams)](JitFuture& /* unused */) mutable {
          // The cost of pre-request check is minimal thanks to
          // std::shared_lock. The cost is in magnitude
          // of 10us.
          auto serverProcessGlobalProfilerStateStackEntryPtr =
              profiler::processglobal::StateStackEntry::current();
          // If server global profiler is enabled, we further pay the
          // cost of thread local profiler state initialization.
          if (serverProcessGlobalProfilerStateStackEntryPtr) {
            // Initialize thread-local profiler state from process-global
            // profiler state.
            enableProfilerLegacy(
                serverProcessGlobalProfilerStateStackEntryPtr->statePtr()
                    ->config());
          }

          auto retFuture = processRpcWithErrors(*rpc, messageType, streams);

          // Response message has been sent at this moment, this post-response
          // work doesn't affect RPC trip time.
          if (serverProcessGlobalProfilerStateStackEntryPtr) {
            // Restore thread-local profiler state.
            thread_event_lists event_lists = disableProfilerLegacy();
            // Put thread_local event_lists into the process-global profiler
            // state.
            profiler::processglobal::pushResultRecursive(
                serverProcessGlobalProfilerStateStackEntryPtr, event_lists);
          }

          return retFuture;
        },
        c10::getCustomClassType<c10::intrusive_ptr<Message>>());

    auto retFutureWithMessageId = retFuture->then(
        [id = request.id()](JitFuture& future) {
          c10::intrusive_ptr<Message> message =
              future.value().toCustomClass<Message>();
          message->setId(id);
          return withStorages(message);
        },
        c10::getCustomClassType<c10::intrusive_ptr<Message>>());

    return retFutureWithMessageId;
  } catch (std::exception& e) {
    rrefContext.clearRecordedPendingRRefsOnError();
    return asFuture(handleError(e, request.type(), request.id()));
  }
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRpcWithErrors(
    RpcCommandBase& rpc,
    const MessageType& messageType,
    const std::vector<c10::Stream>& streams) const {
  try {
    return processRpc(rpc, messageType, streams);
  } catch (std::exception& e) {
    // Pass a dummy message ID since it will be overwritten anyways.
    return asFuture(handleError(e, messageType, -1));
  }
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processScriptCall(
    RpcCommandBase& rpc,
    const std::vector<c10::Stream>& streams) const {
  auto& scriptCall = static_cast<ScriptCall&>(rpc);

  TORCH_CHECK(
      scriptCall.hasOp(), "Only supports the case where ScriptCall has an op");
  auto future =
      runJitOperator(*scriptCall.op(), scriptCall.stackRef(), streams);

  return future->then(
      [](JitFuture& future) {
        return withStorages(ScriptResp(future.value()).toMessage());
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processPythonCall(
    RpcCommandBase& rpc,
    const std::vector<c10::Stream>& /* unused */) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processPythonRemoteCall(
    RpcCommandBase& rpc,
    const std::vector<c10::Stream>& /* unused */) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::assignOwnerRRef(
    const RRefId& rrefId,
    const RRefId& forkId,
    const c10::intrusive_ptr<JitFuture>& valueFuture) const {
  auto& ctx = RRefContext::getInstance();

  c10::intrusive_ptr<OwnerRRef> ownerRRef;
  if (rrefId == forkId) {
    // Creating an owner RRef on self, should already exist in owners map
    ownerRRef =
        fromRRefInterface(ctx.getOwnerRRef(rrefId, /* forceCreated */ true)
                              ->constValue()
                              .toRRef());
  } else {
    ownerRRef = ctx.getOrCreateOwnerRRef(rrefId, valueFuture->elementType());
    // Caller is a user and callee is the owner, add fork
    //
    // NB: rrefId == forkId is true if and only if calling remote to self.
    // In that case both the caller and the callee will access the
    // OwnerRRef. Hence, on the callee side (here), it should not call
    // addForkOfOwner as it is not a fork. To allow callee to distinguish
    // when this request is sent to self, the caller will set forkId using
    // rrefId (OwnerRRef does not have a forkId anyway).
    ctx.addForkOfOwner(rrefId, forkId);
  }

  return valueFuture->then(
      [ownerRRef, rrefId, forkId](JitFuture& future) {
        if (future.hasError()) {
          ownerRRef->setError(future.exception_ptr());
        } else {
          ownerRRef->setValue(future.value());
        }
        return withStorages(RemoteRet(rrefId, forkId).toMessage());
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processScriptRemoteCall(
    RpcCommandBase& rpc,
    const std::vector<c10::Stream>& streams) const {
  auto& scriptRemoteCall = static_cast<ScriptRemoteCall&>(rpc);

  TORCH_CHECK(
      scriptRemoteCall.hasOp(), "ScriptRemoteCall needs to have an op!");
  auto future = runJitOperator(
      *scriptRemoteCall.op(), scriptRemoteCall.stackRef(), streams);

  return assignOwnerRRef(
      scriptRemoteCall.retRRefId(), scriptRemoteCall.retForkId(), future);
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::retrieveOwnerRRef(
    const RRefId& rrefId) const {
  auto& ctx = RRefContext::getInstance();

  auto rrefFuture = ctx.getOwnerRRef(rrefId);

  at::TypePtr type = rrefFuture->elementType();
  TORCH_INTERNAL_ASSERT(type->kind() == at::RRefType::Kind);
  return rrefFuture->thenAsync(
      [](JitFuture& rrefFuture) {
        c10::intrusive_ptr<OwnerRRef> rref =
            fromRRefInterface(rrefFuture.value().toRRef());
        return rref->getFuture();
      },
      type->cast<at::RRefType>()->getElementType());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processScriptRRefFetchCall(RpcCommandBase& rpc) const {
  auto& srf = static_cast<ScriptRRefFetchCall&>(rpc);

  auto future = retrieveOwnerRRef(srf.rrefId());

  return future->then(
      [](JitFuture& future) {
        return withStorages(ScriptRRefFetchRet({future.value()}).toMessage());
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processPythonRRefFetchCall(RpcCommandBase& rpc) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRRefUserDelete(
    RpcCommandBase& rpc) const {
  auto& rud = static_cast<RRefUserDelete&>(rpc);
  auto& ctx = RRefContext::getInstance();
  auto deletedRRef = ctx.delForkOfOwner(rud.rrefId(), rud.forkId());
  handleRRefDelete(deletedRRef);
  return asFuture(RRefAck().toMessage());
}

void RequestCallbackNoPython::handleRRefDelete(
    c10::intrusive_ptr<RRef>& rref) const {
  TORCH_CHECK(!rref->isPyObj(), "RRefs with python objects not supported!");
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRRefChildAccept(
    RpcCommandBase& rpc) const {
  auto& rca = static_cast<RRefChildAccept&>(rpc);
  auto& ctx = RRefContext::getInstance();
  ctx.delPendingChild(rca.forkId());
  return asFuture(RRefAck().toMessage());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRRefForkRequest(
    RpcCommandBase& rpc) const {
  auto& rfr = static_cast<RRefForkRequest&>(rpc);
  auto& ctx = RRefContext::getInstance();
  ctx.addForkOfOwnerIfNotPresent(rfr.rrefId(), rfr.forkId());
  return asFuture(RRefAck().toMessage());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processForwardAutogradReq(
        RpcCommandBase& rpc,
        const std::vector<c10::Stream>& streams) const {
  auto& rpcWithAutograd = static_cast<RpcWithAutograd&>(rpc);

  // Need to reverse the device map for the backward pass of distributed
  // autograd.
  DeviceMap reverseDeviceMap;
  for (const auto& mapEntry : rpcWithAutograd.deviceMap()) {
    reverseDeviceMap.insert({mapEntry.second, mapEntry.first});
  }

  // Attach 'recv' autograd function.
  auto autogradContext = addRecvRpcBackward(
      rpcWithAutograd.autogradMetadata(),
      rpcWithAutograd.tensors(),
      rpcWithAutograd.fromWorkerId(),
      reverseDeviceMap);
  // For this recv thread on server side, before processRpc(),
  // set current_context_id_ to be context_id passed from client.
  // In this way, if there is nested rpc call in python rpc call, original
  // context_id from client can be passed in the chain calls.
  TORCH_INTERNAL_ASSERT(
      autogradContext != nullptr,
      "autogradContext is nullptr, FORWARD_AUTOGRAD_REQ should always get "
      "or create valid autogradContext in addRecvRpcBackward.");

  DistAutogradContextGuard ctxGuard(autogradContext->contextId());

  // Process the original RPC.
  auto wrappedMessageType = rpcWithAutograd.wrappedMessageType();
  // Kick off processing for the nested RPC command.
  // wrappedRpcResponseFuture will be a Future<T> to the result.
  auto wrappedRpcResponseFuture =
      processRpc(rpcWithAutograd.wrappedRpc(), wrappedMessageType, streams);

  auto fromWorkerId = rpcWithAutograd.fromWorkerId();
  // The original future needs to be marked as completed when the wrapped
  // one completes, with the autograd context information wrapped.
  auto responseFuture = wrappedRpcResponseFuture->then(
      [fromWorkerId, ctxId = autogradContext->contextId()](
          JitFuture& wrappedRpcResponseFuture) {
        // As this callback can be invoked by a different thread, we have to
        // make sure that the thread_local states in the previous thread is
        // correctly propagated.
        // NB: The execution of TorchScript functions can also run on a
        // different thread, which is addressed by
        // https://github.com/pytorch/pytorch/pull/36395
        // NB: when adding async UDF support, we should also propagate
        // thread_local states there.
        // TODO: Land on a general solution for RPC ThreadLocalState. See
        // https://github.com/pytorch/pytorch/issues/38510
        DistAutogradContextGuard cbCtxGuard(ctxId);

        if (wrappedRpcResponseFuture.hasError()) {
          // Propagate error to responseFuture if we had one.
          std::rethrow_exception(wrappedRpcResponseFuture.exception_ptr());
        } else {
          auto msg = getMessageWithAutograd(
              fromWorkerId,
              wrappedRpcResponseFuture.value().toCustomClass<Message>(),
              MessageType::FORWARD_AUTOGRAD_RESP);
          return withStorages(std::move(msg));
        }
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());

  return responseFuture;
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processBackwardAutogradReq(
        RpcCommandBase& rpc,
        const std::vector<c10::Stream>& streams) const {
  c10::MultiStreamGuard guard(streams);
  auto& gradientsCall = static_cast<PropagateGradientsReq&>(rpc);
  const auto& autogradMetadata = gradientsCall.getAutogradMetadata();

  // Retrieve the appropriate autograd context.
  auto autogradContext = DistAutogradContainer::getInstance().retrieveContext(
      autogradMetadata.autogradContextId);

  // Lookup the appropriate 'send' function to enqueue.
  std::shared_ptr<SendRpcBackward> sendFunction =
      autogradContext->retrieveSendFunction(autogradMetadata.autogradMessageId);

  // Attach the gradients to the send function.
  sendFunction->setGrads(gradientsCall.getGrads());

  // Now execute the autograd graph using the "distributed engine."
  auto execFuture = DistEngine::getInstance().executeSendFunctionAsync(
      autogradContext, sendFunction, gradientsCall.retainGraph());

  // Our response is satisfied when the rpcs come back.
  return execFuture->then(
      [](JitFuture& execFuture) {
        if (execFuture.hasError()) {
          std::rethrow_exception(execFuture.exception_ptr());
        } else {
          return withStorages(PropagateGradientsResp().toMessage());
        }
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processCleanupAutogradContextReq(RpcCommandBase& rpc) const {
  auto& cleanupContextReq = static_cast<CleanupAutogradContextReq&>(rpc);
  auto cleanupContextId = cleanupContextReq.getContextId();
  // release the context if it still exists on this thread. We need to
  // check if it exists since it may have been deleted by an in-flight
  // RPC. This can create nested RPCs if there are other nodes that get
  // notified to clean up their context.
  DistAutogradContainer::getInstance().releaseContextIfPresent(
      cleanupContextId);
  return asFuture(CleanupAutogradContextResp().toMessage());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processRunWithProfilingReq(RpcCommandBase& rpc) const {
  auto& rpcWithProfilingReq = static_cast<RpcWithProfilingReq&>(rpc);
  auto wrappedMsgType = rpcWithProfilingReq.wrappedMessageType();
  auto profilingConfig = rpcWithProfilingReq.getProfilingConfig();

  if (profilingConfig.state == ProfilerState::KINETO ||
      profilingConfig.state == ProfilerState::KINETO_GPU_FALLBACK ||
      profilingConfig.state == ProfilerState::KINETO_PRIVATEUSE1_FALLBACK) {
    profilingConfig = ProfilerConfig(
        ProfilerState::CPU,
        profilingConfig.report_input_shapes,
        profilingConfig.profile_memory);
  }

  // If requested with CUDA from caller but CUDA is not available on this
  // machine, fallback to CPU and log a warning instead of crashing.
  if (profilingConfig.state == ProfilerState::CUDA && !this->cudaAvailable()) {
    profilingConfig = ProfilerConfig(
        ProfilerState::CPU,
        profilingConfig.report_input_shapes,
        profilingConfig.profile_memory);

    LOG(WARNING) << "Profiler was requested to be enabled with CUDA on this "
                    "node, but CUDA is not available. "
                 << "Falling back to CPU profiling only.";
  }
  TORCH_INTERNAL_ASSERT(
      profilingConfig.state != ProfilerState::CUDA || this->cudaAvailable(),
      "Profiler state set to CUDA but CUDA not available.");
  const auto profilingKeyId = rpcWithProfilingReq.getProfilingId();
  // Enable the profiler with the config from the sender.
  // When enabling on the main thread, ensure profiler states are cleaned
  // up, but defer consolidation of all profiled events to the continuation
  // below.
  ProfilerDisableOptions requestThreadOptions(
      true /* cleanup TLS state */, false /* consolidate events */);
  {
    TLSLegacyProfilerGuard g(
        profilingConfig, std::nullopt, requestThreadOptions);
    TORCH_INTERNAL_ASSERT(
        profilerEnabled(), "Expected profiler to be enabled!");
    // Kick off processing for nested work and get Future<T> result in
    // wrappedRpcResponseFuture
    auto wrappedRpcResponseFuture = processRpc(
        rpcWithProfilingReq.wrappedRpc(),
        wrappedMsgType,
        {}); // TODO: https://github.com/pytorch/pytorch/issues/55757

    auto responseFuture = wrappedRpcResponseFuture->then(
        at::wrapPropagateTLSState([profilingKeyId, profilingConfig](
                                      JitFuture& wrappedRpcResponseFuture) {
          std::vector<LegacyEvent> profiledEvents;
          // Defer consolidation of profiler events until async work has
          // completed (such as async UDF)

          TORCH_INTERNAL_ASSERT(
              profilerEnabled(), "Expected profiler to be enabled!");

          // On continuation thread, don't clean up profiler states, since
          // they will be cleaned up by main thread, and consolidate all
          // events so we obtain asynchronously run events.
          ProfilerDisableOptions opts(false, true);
          auto event_lists = disableProfilerLegacy(opts);
          if (wrappedRpcResponseFuture.hasError()) {
            // Propagate error
            // No need to propagate remote events in the case of an error.
            std::rethrow_exception(wrappedRpcResponseFuture.exception_ptr());
          } else {
            populateRemoteProfiledEvents(
                profiledEvents, profilingConfig, event_lists);
            auto rpcWithProfilingResp = std::make_unique<RpcWithProfilingResp>(
                MessageType::RUN_WITH_PROFILING_RESP,
                wrappedRpcResponseFuture.value().toCustomClass<Message>(),
                profiledEvents,
                profilingKeyId);
            return withStorages(std::move(*rpcWithProfilingResp).toMessage());
          }
        }),
        c10::getCustomClassType<c10::intrusive_ptr<Message>>());

    return responseFuture;
    // Exiting the scope will disable the profiler on this thread with the
    // options specified above.
  }
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRRefBackward(
    RpcCommandBase& rpc) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRpc(
    RpcCommandBase& rpc,
    const MessageType& messageType,
    const std::vector<c10::Stream>& streams) const {
  // TODO: RpcCommandBase should have an abstract execute() method that we can
  // call here instead of having another switch statement here. Even better we
  // could have abstract classes RpcRequest and RpcResp which inherit from
  // RpcCommandBase and RpcRequest declares the abstract method execute() that
  // we can call here. RpcResponse could have an abstract method to convert it
  // to a python object.
  switch (messageType) {
    case MessageType::SCRIPT_CALL: {
      return processScriptCall(rpc, streams);
    }
    case MessageType::PYTHON_CALL: {
      return processPythonCall(rpc, streams);
    }
    case MessageType::SCRIPT_REMOTE_CALL: {
      return processScriptRemoteCall(rpc, streams);
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      return processPythonRemoteCall(rpc, streams);
    }
    case MessageType::SCRIPT_RREF_FETCH_CALL: {
      return processScriptRRefFetchCall(rpc);
    }
    case MessageType::PYTHON_RREF_FETCH_CALL: {
      return processPythonRRefFetchCall(rpc);
    }
    case MessageType::RREF_USER_DELETE: {
      return processRRefUserDelete(rpc);
    }
    case MessageType::RREF_CHILD_ACCEPT: {
      return processRRefChildAccept(rpc);
    }
    case MessageType::RREF_FORK_REQUEST: {
      return processRRefForkRequest(rpc);
    }
    case MessageType::FORWARD_AUTOGRAD_REQ: {
      return processForwardAutogradReq(rpc, streams);
    }
    case MessageType::BACKWARD_AUTOGRAD_REQ: {
      return processBackwardAutogradReq(rpc, streams);
    };
    case MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ: {
      return processCleanupAutogradContextReq(rpc);
    }
    case MessageType::RUN_WITH_PROFILING_REQ: {
      return processRunWithProfilingReq(rpc);
    }
    case MessageType::RREF_BACKWARD_REQ: {
      return processRRefBackward(rpc);
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Request type ", messageType, " not supported.");
    }
  }
}

c10::intrusive_ptr<Message> RequestCallbackNoPython::handleError(
    const std::exception& e,
    const MessageType messageType,
    int64_t messageId) const {
  LOG(ERROR) << "Received error while processing request type " << messageType
             << ": " << e.what();
  // Adding node information to the error here since all processed RPC
  // requests should be going through this function.
  std::string errorMsg = c10::str(
      "Error on Node ",
      DistAutogradContainer::getInstance().getWorkerId(),
      ": ",
      e.what());
  return createExceptionResponse(errorMsg, messageId);
}

bool RequestCallbackNoPython::cudaAvailable() const {
#ifdef USE_CUDA
  return true;
#else
  return false;
#endif
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::runJitOperator(
    const jit::Operator& op,
    std::vector<at::IValue>& stack,
    const std::vector<c10::Stream>& streams) const {
  c10::MultiStreamGuard guard(streams);
  try {
    op.getOperation()(stack);
  } catch (const std::exception&) {
    return asFuture(std::current_exception());
  }
  TORCH_INTERNAL_ASSERT(
      stack.size() == 1,
      "Return value of a builtin operator or a TorchScript function should be "
      "a single IValue, got a vector of size ",
      stack.size());
  TypePtr type = stack.front().type();
  return asFuture(std::move(stack.front()), std::move(type));
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::asFuture(
    IValue value,
    TypePtr type) const {
  auto future = c10::make_intrusive<JitFuture>(
      std::move(type), RpcAgent::getCurrentRpcAgent()->getDevices());
  future->markCompleted(std::move(value));
  return future;
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::asFuture(
    c10::intrusive_ptr<Message> message) const {
  auto future = c10::make_intrusive<JitFuture>(
      at::getCustomClassType<c10::intrusive_ptr<Message>>(),
      RpcAgent::getCurrentRpcAgent()->getDevices());
  std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> storages =
      message->getStorages();
  future->markCompleted(std::move(message), std::move(storages));
  return future;
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::asFuture(
    std::exception_ptr err) const {
  auto future = c10::make_intrusive<JitFuture>(
      at::NoneType::get(), RpcAgent::getCurrentRpcAgent()->getDevices());
  future->setError(std::move(err));
  return future;
}

} // namespace torch::distributed::rpc
