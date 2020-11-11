#include <torch/csrc/distributed/rpc/request_callback_no_python.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h>
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

std::shared_ptr<FutureMessage> RequestCallbackNoPython::processMessage(
    Message& request) const {
  // We need two futures here because it could pause twice when processing a
  // RPC message:
  //  1) waiting for all RRefs in the arguments to become confirmed;
  //  2) waiting for processRpc to finish.
  auto retFuture = std::make_shared<FutureMessage>();
  auto& rrefContext = RRefContext::getInstance();
  try {
    rrefContext.recordThreadLocalPendingRRefs();
    // Deserialize PythonUDF here to trigger RRef unpickling
    std::unique_ptr<RpcCommandBase> rpc = deserializePythonRpcCommand(
        deserializeRequest(request), request.type());
    auto rrefsReadyFuture = rrefContext.waitForThreadLocalPendingRRefs();

    rrefsReadyFuture->addCallback(
        [this,
         retFuture,
         // std::function must be copyable, hence hae to cast the unique_ptr to
         // a shared_ptr here.
         rpc = (std::shared_ptr<RpcCommandBase>)std::move(rpc),
         messageType = request.type(),
         id = request.id()]() {
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

          processRpcWithErrors(*rpc, messageType, id, retFuture);

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
        });
  } catch (std::exception& e) {
    retFuture->markCompleted(handleError(e, request.type(), request.id()));
    rrefContext.clearRecordedPendingRRefsOnError();
  }
  return retFuture;
}

void RequestCallbackNoPython::processRpcWithErrors(
    RpcCommandBase& rpc,
    const MessageType& messageType,
    const int64_t messageId,
    const std::shared_ptr<FutureMessage>& responseFuture) const {
  try {
    processRpc(rpc, messageType, messageId, responseFuture);
  } catch (std::exception& e) {
    responseFuture->markCompleted(handleError(e, messageType, messageId));
  }
}

void RequestCallbackNoPython::processScriptCall(
    ScriptCall& scriptCall,
    const std::function<void(Message)>& markComplete,
    std::vector<at::IValue>& stack,
    const int64_t messageId,
    const std::shared_ptr<FutureMessage>& responseFuture) const {
  TORCH_CHECK(
      scriptCall.hasOp(), "Only supports the case where ScriptCall has an op");
  processScriptCallOp(scriptCall, markComplete, stack);
}

bool RequestCallbackNoPython::processScriptCallOp(
    ScriptCall& scriptCall,
    const std::function<void(Message)>& markComplete,
    std::vector<at::IValue>& stack) const {
  if (scriptCall.hasOp()) {
    scriptCall.op()->getOperation()(&stack);
    TORCH_INTERNAL_ASSERT(
        stack.size() == 1,
        "Return value of a builtin operator or a "
        "TorchScript function should be a single IValue, got a vector of "
        "size ",
        stack.size());
    markComplete(std::move(ScriptResp(std::move(stack.front()))).toMessage());
    return true;
  }
  return false;
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
    const int64_t messageId,
    const std::shared_ptr<FutureMessage>& responseFuture) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

void RequestCallbackNoPython::processPythonRemoteCall(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete,
    const int64_t messageId,
    const std::shared_ptr<FutureMessage>& responseFuture) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

void RequestCallbackNoPython::processRRefBackward(
      RpcCommandBase& rpc,
      const int64_t messageId,
      const std::shared_ptr<FutureMessage>& responseFuture) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

void RequestCallbackNoPython::processScriptRemoteCall(
    ScriptRemoteCall& scriptRemoteCall,
    const std::function<void(void)>& postProcessing,
    std::vector<at::IValue>& stack,
    const c10::intrusive_ptr<OwnerRRef>& ownerRRef) const {
  TORCH_CHECK(
      scriptRemoteCall.hasOp(), "ScriptRemoteCall needs to have an op!");
  processScriptRemoteCallOp(scriptRemoteCall, postProcessing, stack, ownerRRef);
}

bool RequestCallbackNoPython::processScriptRemoteCallOp(
    ScriptRemoteCall& scriptRemoteCall,
    const std::function<void(void)>& postProcessing,
    std::vector<at::IValue>& stack,
    const c10::intrusive_ptr<OwnerRRef>& ownerRRef) const {
  if (scriptRemoteCall.hasOp()) {
    try {
      scriptRemoteCall.op()->getOperation()(&stack);
    } catch (const std::exception& e) {
      // Don't throw in this call, but rather transfer the exception
      // to the rref.
      ownerRRef->setError(std::current_exception());
      postProcessing();
      return true;
    }
    TORCH_INTERNAL_ASSERT(
        stack.size() == 1,
        "Return value of a builtin operator or a "
        "TorchScript function should be a single IValue, got a vector of "
        "size ",
        stack.size());
    ownerRRef->setValue(std::move(stack.front()));
    postProcessing();
    return true;
  }
  return false;
}

void RequestCallbackNoPython::processPythonRRefFetchCall(
    RpcCommandBase& rpc,
    const int64_t messageId,
    const std::shared_ptr<FutureMessage>& responseFuture) const {
  C10_THROW_ERROR(Error, "Python call not supported!");
}

void RequestCallbackNoPython::handleRRefDelete(
    c10::intrusive_ptr<RRef>& rref) const {
  TORCH_CHECK(!rref->isPyObj(), "RRefs with python objects not supported!");
}

void RequestCallbackNoPython::processRpc(
    RpcCommandBase& rpc,
    const MessageType& messageType,
    const int64_t messageId,
    const std::shared_ptr<FutureMessage>& responseFuture) const {
  auto markComplete = [messageId, &responseFuture](Message m) {
    m.setId(messageId);
    responseFuture->markCompleted(std::move(m));
  };

  // TODO: RpcCommandBase should have an abstract execute() method that we can
  // call here instead of having another switch statement here. Even better we
  // could have abstract classes RpcRequest and RpcResp which inherit from
  // RpcCommandBase and RpcRequest declares the abstract method execute() that
  // we can call here. RpcResponse could have an abstract method to convert it
  // to a python object.
  switch (messageType) {
    case MessageType::SCRIPT_CALL: {
      auto& scriptCall = static_cast<ScriptCall&>(rpc);

      // scriptCall is only alive within this block, use reference to avoid copy
      auto& stack = scriptCall.stackRef();
      processScriptCall(
          scriptCall, markComplete, stack, messageId, responseFuture);
      return;
    }
    case MessageType::PYTHON_CALL: {
      processPythonCall(rpc, markComplete, messageId, responseFuture);
      return;
    }
    case MessageType::SCRIPT_REMOTE_CALL: {
      auto& scriptRemoteCall = static_cast<ScriptRemoteCall&>(rpc);
      auto rrefId = scriptRemoteCall.retRRefId();
      auto forkId = scriptRemoteCall.retForkId();
      auto& ctx = RRefContext::getInstance();

      auto postProcessing = [rrefId, forkId, messageId, responseFuture]() {
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
        m.setId(messageId);
        responseFuture->markCompleted(std::move(m));
      };

      // scriptRemoteCall is only alive within this block, use reference to
      // avoid copy. If the underlying code runs with a continuation, runAsync()
      // below will std::move the appropriate portion of the stack.
      TypePtr returnType = getScriptRemoteCallType(scriptRemoteCall);
      c10::intrusive_ptr<OwnerRRef> ownerRRef;
      if (rrefId == forkId) {
        // Creating an owner RRef on self, should already exist in owners map
        ownerRRef =
            ctx.getOwnerRRef(rrefId, /* forceCreated */ true)->constValue();
      } else {
        ownerRRef = ctx.getOrCreateOwnerRRef(rrefId, returnType);
      }

      auto& stack = scriptRemoteCall.stackRef();
      processScriptRemoteCall(
          scriptRemoteCall, postProcessing, stack, ownerRRef);
      return;
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      processPythonRemoteCall(rpc, markComplete, messageId, responseFuture);
      return;
    }
    case MessageType::SCRIPT_RREF_FETCH_CALL: {
      auto& srf = static_cast<ScriptRRefFetchCall&>(rpc);
      auto& ctx = RRefContext::getInstance();

      auto futureOwner = ctx.getOwnerRRef(srf.rrefId());

      if (futureOwner->completed()) { // optional fast-path
        // the OwnerRRef has been created
        const auto& rref = futureOwner->constValue();
        if (rref->hasValue()) {
          markComplete(ScriptRRefFetchRet({rref->getValue()}).toMessage());
          return;
        }
      }

      futureOwner->addCallback([responseFuture, messageId, futureOwner]() {
        const auto& rref = futureOwner->constValue();
        auto whenValueSet = rref->getFuture();

        // Our response is satisfied when the rpc.remote() request
        // finishes executing on the owner.
        whenValueSet->addCallback(
            [responseFuture, messageId, rref, whenValueSet]() {
              if (whenValueSet->hasError()) {
                responseFuture->setError(
                    whenValueSet->tryRetrieveErrorMessage());
                return;
              }
              try {
                Message m = ScriptRRefFetchRet({rref->getValue()}).toMessage();
                m.setId(messageId);
                responseFuture->markCompleted(std::move(m));
              } catch (const std::exception& e) {
                responseFuture->setError(e.what());
              }
            });
      });

      return;
    }
    case MessageType::PYTHON_RREF_FETCH_CALL: {
      processPythonRRefFetchCall(rpc, messageId, responseFuture);
      return;
    }
    case MessageType::RREF_USER_DELETE: {
      auto& rud = static_cast<RRefUserDelete&>(rpc);
      auto& ctx = RRefContext::getInstance();
      auto deletedRRef = ctx.delForkOfOwner(rud.rrefId(), rud.forkId());
      handleRRefDelete(deletedRRef);
      markComplete(std::move(RRefAck()).toMessage());
      return;
    }
    case MessageType::RREF_CHILD_ACCEPT: {
      auto& rca = static_cast<RRefChildAccept&>(rpc);
      auto& ctx = RRefContext::getInstance();
      ctx.delPendingChild(rca.forkId());
      markComplete(std::move(RRefAck()).toMessage());
      return;
    }
    case MessageType::RREF_FORK_REQUEST: {
      auto& rfr = static_cast<RRefForkRequest&>(rpc);
      auto& ctx = RRefContext::getInstance();
      ctx.addForkOfOwnerIfNotPresent(rfr.rrefId(), rfr.forkId());
      markComplete(RRefAck().toMessage());
      return;
    }
    case MessageType::FORWARD_AUTOGRAD_REQ: {
      auto& rpcWithAutograd = static_cast<RpcWithAutograd&>(rpc);

      // Attach 'recv' autograd function.
      auto autogradContext = addRecvRpcBackward(
          rpcWithAutograd.autogradMetadata(),
          rpcWithAutograd.tensors(),
          rpcWithAutograd.fromWorkerId());
      // For this recv thread on server side, before processRpc(),
      // set current_context_id_ to be context_id passed from client.
      // In this way, if there is nested rpc call in python rpc call, original
      // context_id from client can be passed in the chain calls.
      auto& autogradContainer = DistAutogradContainer::getInstance();
      TORCH_INTERNAL_ASSERT(
          autogradContext != nullptr,
          "autogradContext is nullptr, FORWARD_AUTOGRAD_REQ should always get "
          "or create valid autogradContext in addRecvRpcBackward.");

      DistAutogradContextGuard ctxGuard(autogradContext->contextId());

      // Process the original RPC.
      auto wrappedMessageType = rpcWithAutograd.wrappedMessageType();
      // Make an overall future for the wrapped response.
      auto wrappedRpcResponseFuture = std::make_shared<FutureMessage>();
      // Kick off processing for the nested RPC command.
      // wrappedRpcResponseFuture will be a Future<T> to the result.
      processRpc(
          rpcWithAutograd.wrappedRpc(),
          wrappedMessageType,
          messageId,
          wrappedRpcResponseFuture);

      auto fromWorkerId = rpcWithAutograd.fromWorkerId();
      // The original future needs to be marked as completed when the wrapped
      // one completes, with the autograd context information wrapped.
      // Uses weak_ptr so we can std::move the value.
      wrappedRpcResponseFuture->addCallback(
          [responseFuture,
           messageId,
           fromWorkerId,
           weak = std::weak_ptr<FutureMessage>(wrappedRpcResponseFuture),
           ctxId = autogradContext->contextId()]() {
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

            auto wrappedRpcResponseFuture = weak.lock();
            TORCH_INTERNAL_ASSERT(wrappedRpcResponseFuture);
            if (wrappedRpcResponseFuture->hasError()) {
              // Propagate error to responseFuture if we had one.
              responseFuture->setError(
                  wrappedRpcResponseFuture->error()->what());
            } else {
              auto msg = getMessageWithAutograd(
                  fromWorkerId,
                  std::move(*wrappedRpcResponseFuture).moveValue(),
                  MessageType::FORWARD_AUTOGRAD_RESP);
              msg.setId(messageId);
              responseFuture->markCompleted(std::move(msg));
            }
          });
      return;
    }
    case MessageType::BACKWARD_AUTOGRAD_REQ: {
      auto& gradientsCall = static_cast<PropagateGradientsReq&>(rpc);
      const auto& autogradMetadata = gradientsCall.getAutogradMetadata();

      // Retrieve the appropriate autograd context.
      auto autogradContext =
          DistAutogradContainer::getInstance().retrieveContext(
              autogradMetadata.autogradContextId);

      // Lookup the appropriate 'send' function to enqueue.
      std::shared_ptr<SendRpcBackward> sendFunction =
          autogradContext->retrieveSendFunction(
              autogradMetadata.autogradMessageId);

      // Attach the gradients to the send function.
      sendFunction->setGrads(gradientsCall.getGrads());

      // Now execute the autograd graph using the "distributed engine."
      auto execFuture = DistEngine::getInstance().executeSendFunctionAsync(
          autogradContext, sendFunction, gradientsCall.retainGraph());

      // Our response is satisfied when the rpcs come back.
      execFuture->addCallback([responseFuture, messageId, execFuture]() {
        if (!execFuture->hasError()) {
          Message m = std::move(PropagateGradientsResp()).toMessage();
          m.setId(messageId);
          responseFuture->markCompleted(std::move(m));
        } else {
          responseFuture->setError(execFuture->tryRetrieveErrorMessage());
        }
      });
      return;
    };
    case MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ: {
      auto& cleanupContextReq = static_cast<CleanupAutogradContextReq&>(rpc);
      auto cleanupContextId = cleanupContextReq.getContextId();
      // release the context if it still exists on this thread. We need to
      // check if it exists since it may have been deleted by an in-flight
      // RPC. This can create nested RPCs if there are other nodes that get
      // notified to clean up their context.
      DistAutogradContainer::getInstance().releaseContextIfPresent(
          cleanupContextId);
      markComplete(std::move(CleanupAutogradContextResp()).toMessage());
      return;
    }
    case MessageType::RUN_WITH_PROFILING_REQ: {
      auto& rpcWithProfilingReq = static_cast<RpcWithProfilingReq&>(rpc);
      auto wrappedMsgType = rpcWithProfilingReq.wrappedMessageType();
      auto profilingConfig = rpcWithProfilingReq.getProfilingConfig();
      // If requested with CUDA from caller but CUDA is not available on this
      // machine, fallback to CPU and log a warning instead of crashing.
      if (profilingConfig.state ==
              torch::autograd::profiler::ProfilerState::CUDA &&
          !this->cudaAvailable()) {
        profilingConfig = torch::autograd::profiler::ProfilerConfig(
            torch::autograd::profiler::ProfilerState::CPU,
            profilingConfig.report_input_shapes,
            profilingConfig.profile_memory);

        LOG(WARNING)
            << "Profiler was requested to be enabled with CUDA on this node, but CUDA is not available. "
            << "Falling back to CPU profiling only.";
      }
      TORCH_INTERNAL_ASSERT(
          profilingConfig.state !=
                  torch::autograd::profiler::ProfilerState::CUDA ||
              this->cudaAvailable(),
          "Profiler state set to CUDA but CUDA not available.");
      const auto profilingKeyId = rpcWithProfilingReq.getProfilingId();
      auto wrappedRpcResponseFuture = std::make_shared<FutureMessage>();
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
        processRpc(
            rpcWithProfilingReq.wrappedRpc(),
            wrappedMsgType,
            messageId,
            wrappedRpcResponseFuture);

        wrappedRpcResponseFuture->addCallback(
            at::wrapPropagateTLSState<void>([wrappedRpcResponseFuture,
                                             responseFuture,
                                             profilingKeyId,
                                             profilingConfig] {
              std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents;
              // Defer consolidation of profiler events until async work has
              // completed (such as async UDF)

              TORCH_INTERNAL_ASSERT(
                  torch::autograd::profiler::profilerEnabled(),
                  "Expected profiler to be enabled!");

              // On continuation thread, don't clean up profiler states, since
              // they will be cleaned up by main thread, and consolidate all
              // events so we obtain asynchronously run events.
              torch::autograd::profiler::ProfilerDisableOptions opts(
                  false, true);
              auto event_lists =
                  torch::autograd::profiler::disableProfilerLegacy(opts);
              if (wrappedRpcResponseFuture->hasError()) {
                // Propagate error
                // No need to propagate remote events in the case of an error.
                responseFuture->setError(
                    wrappedRpcResponseFuture->error()->what());
              } else {
                populateRemoteProfiledEvents(
                    profiledEvents, profilingConfig, event_lists);
                auto rpcWithProfilingResp =
                    std::make_unique<RpcWithProfilingResp>(
                        MessageType::RUN_WITH_PROFILING_RESP,
                        std::move(*wrappedRpcResponseFuture).moveValue(),
                        profiledEvents,
                        profilingKeyId);
                responseFuture->markCompleted(
                    std::move(*rpcWithProfilingResp).toMessage());
              }
            }));
        // Exiting the scope will disable the profiler on this thread with the
        // options specified above.
      }
      return;
    }
    case MessageType::RREF_BACKWARD_REQ: {
      processRRefBackward(rpc, messageId, responseFuture);
      return;
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Request type ", messageType, " not supported.");
    }
  }
}

Message RequestCallbackNoPython::handleError(
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

} // namespace rpc
} // namespace distributed
} // namespace torch
