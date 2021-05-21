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

namespace torch {
namespace distributed {
namespace rpc {

using namespace torch::distributed::autograd;

// When request message has autograd info, processMessage() will set up valid
// current context id properly. This struct is used to clean up current context
// id after processMessage() is done.
struct DistAutogradContextGuard {
  explicit DistAutogradContextGuard(int64_t ctxId) {
    auto& container = DistAutogradContainer::getInstance();
    prevCtxId_ = container.currentContextId();
    container.forceCurrentContextId(ctxId);
  }
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
    std::shared_ptr<LazyStreamContext> ctx) const {
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
         // std::function must be copyable, hence hae to cast the unique_ptr to
         // a shared_ptr here.
         rpc = (std::shared_ptr<RpcCommandBase>)std::move(rpc),
         messageType = request.type(),
         ctx = std::move(ctx)](JitFuture& /* unused */) mutable {
          c10::MultiStreamGuard guard(
              ctx ? ctx->getReservedStreams() : ArrayRef<Stream>({}));
          // The cost of pre-request check is minimal thanks to
          // std::shared_lock. The cost is in magnitude
          // of 10us.
          auto serverProcessGlobalProfilerStateStackEntryPtr =
              profiler::processglobal::StateStackEntry::current();
          // If server global profiler is enabled, we futher pay the
          // cost of thread local profiler state initialization.
          if (serverProcessGlobalProfilerStateStackEntryPtr) {
            // Initialize thread-local profiler state from process-global
            // profiler state.
            ::torch::autograd::profiler::enableProfilerLegacy(
                serverProcessGlobalProfilerStateStackEntryPtr->statePtr()
                    ->config());
          }

          auto retFuture =
              processRpcWithErrors(*rpc, messageType, std::move(ctx));

          // Response message has been sent at this moment, this post-response
          // work doesn't affect RPC trip time.
          if (serverProcessGlobalProfilerStateStackEntryPtr) {
            // Restore thread-local profiler state.
            ::torch::autograd::profiler::thread_event_lists event_lists =
                ::torch::autograd::profiler::disableProfilerLegacy();
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
          return message;
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
    std::shared_ptr<LazyStreamContext> ctx) const {
  try {
    return processRpc(rpc, messageType, std::move(ctx));
  } catch (std::exception& e) {
    // Pass a dummy message ID since it will be overwritten anyways.
    return asFuture(handleError(e, messageType, -1));
  }
}

void RequestCallbackNoPython::processScriptCall(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete,
    const c10::intrusive_ptr<JitFuture>& /* unused */) const {
  auto& scriptCall = static_cast<ScriptCall&>(rpc);
  auto& stack = scriptCall.stackRef();
  TORCH_CHECK(
      scriptCall.hasOp(), "Only supports the case where ScriptCall has an op");
  processScriptCallOp(scriptCall, markComplete, stack);
}

void RequestCallbackNoPython::processScriptCallOp(
    ScriptCall& scriptCall,
    const std::function<void(Message)>& markComplete,
    std::vector<at::IValue>& stack) const {
  TORCH_INTERNAL_ASSERT(scriptCall.hasOp());
  auto future = runJitOperator(*scriptCall.op(), stack);
  future->addCallback([markComplete](JitFuture& future) {
    markComplete(ScriptResp(future.value()).toMessage());
  });
}

TypePtr RequestCallbackNoPython::getScriptRemoteCallType(
    ScriptRemoteCall& scriptRemoteCall) const {
  TORCH_CHECK(
      scriptRemoteCall.hasOp(),
      "Only supports the case where ScriptCall has an op");
  return scriptRemoteCall.op()->schema().returns()[0].type();
}

void RequestCallbackNoPython::processPythonCall(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete,
    const c10::intrusive_ptr<JitFuture>& /* unused */) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

void RequestCallbackNoPython::processPythonRemoteCall(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete,
    const c10::intrusive_ptr<JitFuture>& /* unused */,
    std::shared_ptr<LazyStreamContext> /* unused */) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processScriptRemoteCall(
    ScriptRemoteCall& scriptRemoteCall,
    std::vector<at::IValue>& stack) const {
  TORCH_CHECK(
      scriptRemoteCall.hasOp(), "ScriptRemoteCall needs to have an op!");
  return processScriptRemoteCallOp(scriptRemoteCall, stack);
}

void RequestCallbackNoPython::processBaseScriptRemoteCall(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete,
    const c10::intrusive_ptr<JitFuture>& responseFuture) const {
  auto& scriptRemoteCall = static_cast<ScriptRemoteCall&>(rpc);
  auto rrefId = scriptRemoteCall.retRRefId();
  auto forkId = scriptRemoteCall.retForkId();
  auto& ctx = RRefContext::getInstance();

  auto postProcessing = [rrefId, forkId, responseFuture]() {
    if (rrefId != forkId) {
      // Caller is a user and callee is the owner, add fork
      //
      // NB: rrefId == forkId is true if and only if calling remote to
      // self. In that case both the caller and the callee will access
      // the OwnerRRef. Hence, on the callee side (here), it should not
      // call addForkOfOwner as it is not a fork. To allow callee to
      // distinguish when this request is sent to self, the caller will
      // set forkId using rrefId (OwnerRRef does not have a forkId
      // anyway).
      RRefContext::getInstance().addForkOfOwner(rrefId, forkId);
    }
    Message m = RemoteRet(rrefId, forkId).toMessage();
    responseFuture->markCompleted(
        IValue(c10::make_intrusive<Message>(std::move(m))));
  };

  // scriptRemoteCall is only alive within this block, use reference to
  // avoid copy. If the underlying code runs with a continuation, runAsync()
  // below will std::move the appropriate portion of the stack.
  TypePtr returnType = getScriptRemoteCallType(scriptRemoteCall);
  c10::intrusive_ptr<OwnerRRef> ownerRRef;
  if (rrefId == forkId) {
    // Creating an owner RRef on self, should already exist in owners map
    ownerRRef =
        fromRRefInterface(ctx.getOwnerRRef(rrefId, /* forceCreated */ true)
                              ->constValue()
                              .toRRef());
  } else {
    ownerRRef = ctx.getOrCreateOwnerRRef(rrefId, returnType);
  }

  auto& stack = scriptRemoteCall.stackRef();
  auto jitFuture = processScriptRemoteCall(scriptRemoteCall, stack);

  jitFuture->addCallback([postProcessing, ownerRRef](JitFuture& future) {
    if (future.hasError()) {
      ownerRRef->setError(future.exception_ptr());
    } else {
      ownerRRef->setValue(future.value());
    }
    postProcessing();
  });
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processScriptRemoteCallOp(
        ScriptRemoteCall& scriptRemoteCall,
        std::vector<at::IValue>& stack) const {
  TORCH_INTERNAL_ASSERT(scriptRemoteCall.hasOp());
  return runJitOperator(*scriptRemoteCall.op(), stack);
}

void RequestCallbackNoPython::processScriptRRefFetchCall(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete,
    const c10::intrusive_ptr<JitFuture>& responseFuture) const {
  auto& srf = static_cast<ScriptRRefFetchCall&>(rpc);
  auto& ctx = RRefContext::getInstance();

  auto futureOwner = ctx.getOwnerRRef(srf.rrefId());

  if (futureOwner->completed()) { // optional fast-path
    // the OwnerRRef has been created
    const auto& rref = fromRRefInterface(futureOwner->constValue().toRRef());
    if (rref->hasValue()) {
      markComplete(ScriptRRefFetchRet({rref->getValue()}).toMessage());
      return;
    }
  }

  futureOwner->addCallback([responseFuture](JitFuture& futureOwner) {
    const auto& rref = fromRRefInterface(futureOwner.constValue().toRRef());
    auto whenValueSet = rref->getFuture();

    // Our response is satisfied when the rpc.remote() request
    // finishes executing on the owner.
    whenValueSet->addCallback([responseFuture, rref](JitFuture& whenValueSet) {
      if (whenValueSet.hasError()) {
        responseFuture->setError(whenValueSet.exception_ptr());
        return;
      }
      try {
        Message m = ScriptRRefFetchRet({rref->getValue()}).toMessage();
        responseFuture->markCompleted(
            IValue(c10::make_intrusive<Message>(std::move(m))));
      } catch (const std::exception& /* unused */) {
        responseFuture->setError(std::current_exception());
      }
    });
  });
}

void RequestCallbackNoPython::processPythonRRefFetchCall(
    RpcCommandBase& rpc,
    const c10::intrusive_ptr<JitFuture>& /* unused */,
    std::shared_ptr<LazyStreamContext> /* unused */) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

void RequestCallbackNoPython::processRRefUserDelete(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete) const {
  auto& rud = static_cast<RRefUserDelete&>(rpc);
  auto& ctx = RRefContext::getInstance();
  auto deletedRRef = ctx.delForkOfOwner(rud.rrefId(), rud.forkId());
  handleRRefDelete(deletedRRef);
  markComplete(std::move(RRefAck()).toMessage());
}

void RequestCallbackNoPython::handleRRefDelete(
    c10::intrusive_ptr<RRef>& rref) const {
  TORCH_CHECK(!rref->isPyObj(), "RRefs with python objects not supported!");
}

void RequestCallbackNoPython::processRRefChildAccept(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete) const {
  auto& rca = static_cast<RRefChildAccept&>(rpc);
  auto& ctx = RRefContext::getInstance();
  ctx.delPendingChild(rca.forkId());
  markComplete(std::move(RRefAck()).toMessage());
}

void RequestCallbackNoPython::processRRefForkRequest(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete) const {
  auto& rfr = static_cast<RRefForkRequest&>(rpc);
  auto& ctx = RRefContext::getInstance();
  ctx.addForkOfOwnerIfNotPresent(rfr.rrefId(), rfr.forkId());
  markComplete(RRefAck().toMessage());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processForwardAutogradReq(
        RpcCommandBase& rpc,
        std::shared_ptr<LazyStreamContext> ctx) const {
  auto& rpcWithAutograd = static_cast<RpcWithAutograd&>(rpc);

  // Need to reverse the device map for the backward pass of distributed
  // autograd.
  std::unordered_map<c10::Device, c10::Device> reverseDeviceMap;
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
  auto wrappedRpcResponseFuture = processRpc(
      rpcWithAutograd.wrappedRpc(), wrappedMessageType, std::move(ctx));

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
              std::move(
                  *wrappedRpcResponseFuture.value().toCustomClass<Message>()),
              MessageType::FORWARD_AUTOGRAD_RESP);
          return c10::make_intrusive<Message>(std::move(msg));
        }
      },
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());

  return responseFuture;
}

void RequestCallbackNoPython::processBackwardAutogradReq(
    RpcCommandBase& rpc,
    const c10::intrusive_ptr<JitFuture>& responseFuture) const {
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
  execFuture->addCallback([responseFuture](JitFuture& execFuture) {
    if (!execFuture.hasError()) {
      Message m = std::move(PropagateGradientsResp()).toMessage();
      responseFuture->markCompleted(
          IValue(c10::make_intrusive<Message>(std::move(m))));
    } else {
      responseFuture->setError(execFuture.exception_ptr());
    }
  });
}

void RequestCallbackNoPython::processCleanupAutogradContextReq(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete) const {
  auto& cleanupContextReq = static_cast<CleanupAutogradContextReq&>(rpc);
  auto cleanupContextId = cleanupContextReq.getContextId();
  // release the context if it still exists on this thread. We need to
  // check if it exists since it may have been deleted by an in-flight
  // RPC. This can create nested RPCs if there are other nodes that get
  // notified to clean up their context.
  DistAutogradContainer::getInstance().releaseContextIfPresent(
      cleanupContextId);
  markComplete(std::move(CleanupAutogradContextResp()).toMessage());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::
    processRunWithProfilingReq(RpcCommandBase& rpc) const {
  auto& rpcWithProfilingReq = static_cast<RpcWithProfilingReq&>(rpc);
  auto wrappedMsgType = rpcWithProfilingReq.wrappedMessageType();
  auto profilingConfig = rpcWithProfilingReq.getProfilingConfig();
  // If requested with CUDA from caller but CUDA is not available on this
  // machine, fallback to CPU and log a warning instead of crashing.
  if (profilingConfig.state == torch::autograd::profiler::ProfilerState::CUDA &&
      !this->cudaAvailable()) {
    profilingConfig = torch::autograd::profiler::ProfilerConfig(
        torch::autograd::profiler::ProfilerState::CPU,
        profilingConfig.report_input_shapes,
        profilingConfig.profile_memory);

    LOG(WARNING) << "Profiler was requested to be enabled with CUDA on this "
                    "node, but CUDA is not available. "
                 << "Falling back to CPU profiling only.";
  }
  TORCH_INTERNAL_ASSERT(
      profilingConfig.state != torch::autograd::profiler::ProfilerState::CUDA ||
          this->cudaAvailable(),
      "Profiler state set to CUDA but CUDA not available.");
  const auto profilingKeyId = rpcWithProfilingReq.getProfilingId();
  // Enable the profiler with the config from the sender.
  // When enabling on the main thread, ensure profiler states are cleaned
  // up, but defer consolidation of all profiled events to the continuation
  // below.
  torch::autograd::profiler::ProfilerDisableOptions requestThreadOptions(
      true /* cleanup TLS state */, false /* consolidate events */);
  {
    torch::autograd::profiler::TLSProfilerGuard g(
        profilingConfig, c10::nullopt, requestThreadOptions);
    TORCH_INTERNAL_ASSERT(
        torch::autograd::profiler::profilerEnabled(),
        "Expected profiler to be enabled!");
    // Kick off processing for nested work and get Future<T> result in
    // wrappedRpcResponseFuture
    auto wrappedRpcResponseFuture = processRpc(
        rpcWithProfilingReq.wrappedRpc(),
        wrappedMsgType,
        {}); // TODO: https://github.com/pytorch/pytorch/issues/55757

    auto responseFuture = wrappedRpcResponseFuture->then(
        at::wrapPropagateTLSState([profilingKeyId, profilingConfig](
                                      JitFuture& wrappedRpcResponseFuture) {
          std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents;
          // Defer consolidation of profiler events until async work has
          // completed (such as async UDF)

          TORCH_INTERNAL_ASSERT(
              torch::autograd::profiler::profilerEnabled(),
              "Expected profiler to be enabled!");

          // On continuation thread, don't clean up profiler states, since
          // they will be cleaned up by main thread, and consolidate all
          // events so we obtain asynchronously run events.
          torch::autograd::profiler::ProfilerDisableOptions opts(false, true);
          auto event_lists =
              torch::autograd::profiler::disableProfilerLegacy(opts);
          if (wrappedRpcResponseFuture.hasError()) {
            // Propagate error
            // No need to propagate remote events in the case of an error.
            std::rethrow_exception(wrappedRpcResponseFuture.exception_ptr());
          } else {
            populateRemoteProfiledEvents(
                profiledEvents, profilingConfig, event_lists);
            auto rpcWithProfilingResp = std::make_unique<RpcWithProfilingResp>(
                MessageType::RUN_WITH_PROFILING_RESP,
                std::move(
                    *wrappedRpcResponseFuture.value().toCustomClass<Message>()),
                profiledEvents,
                profilingKeyId);
            return c10::make_intrusive<Message>(
                std::move(*rpcWithProfilingResp).toMessage());
          }
        }),
        c10::getCustomClassType<c10::intrusive_ptr<Message>>());

    return responseFuture;
    // Exiting the scope will disable the profiler on this thread with the
    // options specified above.
  }
}

void RequestCallbackNoPython::processRRefBackward(
    RpcCommandBase& rpc,
    const c10::intrusive_ptr<JitFuture>& /* unused */) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::processRpc(
    RpcCommandBase& rpc,
    const MessageType& messageType,
    std::shared_ptr<LazyStreamContext> ctx) const {
  // TODO Avoid creating a future here and passing it down, and instead allow
  // each method to create the future however it wants and pass it back up.
  auto responseFuture = c10::make_intrusive<JitFuture>(
      c10::getCustomClassType<c10::intrusive_ptr<Message>>());
  auto markComplete = [&responseFuture](Message m) {
    responseFuture->markCompleted(
        IValue(c10::make_intrusive<Message>(std::move(m))));
  };
  // TODO: RpcCommandBase should have an abstract execute() method that we can
  // call here instead of having another switch statement here. Even better we
  // could have abstract classes RpcRequest and RpcResp which inherit from
  // RpcCommandBase and RpcRequest declares the abstract method execute() that
  // we can call here. RpcResponse could have an abstract method to convert it
  // to a python object.
  switch (messageType) {
    case MessageType::SCRIPT_CALL: {
      processScriptCall(rpc, markComplete, responseFuture);
      return responseFuture;
    }
    case MessageType::PYTHON_CALL: {
      processPythonCall(rpc, markComplete, responseFuture);
      return responseFuture;
    }
    case MessageType::SCRIPT_REMOTE_CALL: {
      processBaseScriptRemoteCall(rpc, markComplete, responseFuture);
      return responseFuture;
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      processPythonRemoteCall(
          rpc, markComplete, responseFuture, std::move(ctx));
      return responseFuture;
    }
    case MessageType::SCRIPT_RREF_FETCH_CALL: {
      processScriptRRefFetchCall(rpc, markComplete, responseFuture);
      return responseFuture;
    }
    case MessageType::PYTHON_RREF_FETCH_CALL: {
      processPythonRRefFetchCall(rpc, responseFuture, std::move(ctx));
      return responseFuture;
    }
    case MessageType::RREF_USER_DELETE: {
      processRRefUserDelete(rpc, markComplete);
      return responseFuture;
    }
    case MessageType::RREF_CHILD_ACCEPT: {
      processRRefChildAccept(rpc, markComplete);
      return responseFuture;
    }
    case MessageType::RREF_FORK_REQUEST: {
      processRRefForkRequest(rpc, markComplete);
      return responseFuture;
    }
    case MessageType::FORWARD_AUTOGRAD_REQ: {
      return processForwardAutogradReq(rpc, std::move(ctx));
    }
    case MessageType::BACKWARD_AUTOGRAD_REQ: {
      processBackwardAutogradReq(rpc, responseFuture);
      return responseFuture;
    };
    case MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ: {
      processCleanupAutogradContextReq(rpc, markComplete);
      return responseFuture;
    }
    case MessageType::RUN_WITH_PROFILING_REQ: {
      return processRunWithProfilingReq(rpc);
    }
    case MessageType::RREF_BACKWARD_REQ: {
      processRRefBackward(rpc, responseFuture);
      return responseFuture;
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
  return c10::make_intrusive<Message>(
      createExceptionResponse(errorMsg, messageId));
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
    std::vector<at::IValue>& stack) const {
  try {
    op.getOperation()(&stack);
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
  auto future = c10::make_intrusive<JitFuture>(std::move(type));
  future->markCompleted(std::move(value));
  return future;
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::asFuture(
    c10::intrusive_ptr<Message> message) const {
  return asFuture(
      std::move(message),
      at::getCustomClassType<c10::intrusive_ptr<Message>>());
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::asFuture(
    Message message) const {
  return asFuture(c10::make_intrusive<Message>(std::move(message)));
}

c10::intrusive_ptr<JitFuture> RequestCallbackNoPython::asFuture(
    std::exception_ptr err) const {
  auto future = c10::make_intrusive<JitFuture>(at::NoneType::get());
  future->setError(err);
  return future;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
