#include <torch/csrc/distributed/rpc/request_callback_impl.h>

#include <ATen/ThreadLocalState.h>
#include <c10/util/C++17.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/context/context.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/python_call.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_resp.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/distributed/rpc/unpickled_python_call.h>
#include <torch/csrc/distributed/rpc/unpickled_python_remote_call.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace distributed {
namespace rpc {

using namespace torch::distributed::autograd;

namespace {

std::unique_ptr<RpcCommandBase> deserializePythonRpcCommandReference(
    RpcCommandBase& rpc,
    const MessageType& messageType) {
  switch (messageType) {
    case MessageType::PYTHON_CALL: {
      auto& pc = static_cast<PythonCall&>(rpc);
      return std::make_unique<UnpickledPythonCall>(pc.serializedPyObj());
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      auto& prc = static_cast<PythonRemoteCall&>(rpc);
      return std::make_unique<UnpickledPythonRemoteCall>(
          prc.serializedPyObj(), prc.retRRefId(), prc.retForkId());
    }
    case MessageType::FORWARD_AUTOGRAD_REQ: {
      // Deserialize the wrapped RPC if it contains Python UDF
      auto& rwa = static_cast<RpcWithAutograd&>(rpc);
      auto& wrappedRpc = rwa.wrappedRpc();
      auto pythonRpc = deserializePythonRpcCommandReference(
          wrappedRpc, rwa.wrappedMessageType());
      if (pythonRpc) {
        rwa.setWrappedRpc(std::move(pythonRpc));
      }
      return nullptr;
    }
    default: {
      return nullptr;
    }
  }
}

std::unique_ptr<RpcCommandBase> deserializePythonRpcCommand(
    std::unique_ptr<RpcCommandBase> rpc,
    const MessageType& messageType) {
  auto pythonRpc = deserializePythonRpcCommandReference(*rpc, messageType);
  return pythonRpc ? std::move(pythonRpc) : std::move(rpc);
}

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

} // anonymous namespace

Message RequestCallbackImpl::handleError(
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

void RequestCallbackImpl::processRpc(
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
      if (scriptCall.hasOp()) {
        scriptCall.op()->getOperation()(stack);
        TORCH_INTERNAL_ASSERT(
            stack.size() == 1,
            "Return value of a builtin operator or a "
            "TorchScript function should be a single IValue, got a vector of "
            "size ",
            stack.size());
        markComplete(
            std::move(ScriptResp(std::move(stack.front()))).toMessage());
        return;
      }

      // runAsync() starts in the calling thread, but may return an uncompleted
      // future (though for non-async code, it will typically be completed).
      // If it was async, our callback will typically be invoked by the
      // continuation on an at::launch() thread.
      auto jitFuture = PythonRpcHandler::getInstance()
                           .jitCompilationUnit()
                           ->get_function(scriptCall.qualifiedName())
                           .runAsync(stack);

      if (jitFuture->completed()) {
        markComplete(
            std::move(ScriptResp(std::move(jitFuture->value()))).toMessage());
        return;
      }
      jitFuture->addCallback([responseFuture, messageId, jitFuture]() {
        try {
          Message m = ScriptResp(std::move(jitFuture->value())).toMessage();
          m.setId(messageId);
          responseFuture->markCompleted(std::move(m));
        } catch (const std::exception& e) {
          responseFuture->setError(e.what());
        }
      });
      return;
    }
    case MessageType::PYTHON_CALL: {
      auto& upc = static_cast<UnpickledPythonCall&>(rpc);
      auto& pythonRpcHandler = PythonRpcHandler::getInstance();
      std::shared_ptr<SerializedPyObj> serializedPyObj = nullptr;
      {
        py::gil_scoped_acquire acquire;
        serializedPyObj =
            std::make_shared<SerializedPyObj>(pythonRpcHandler.serialize(
                pythonRpcHandler.runPythonUdf(std::move(upc).movePythonUdf())));
      }
      markComplete(
          std::move(PythonResp(std::move(*serializedPyObj))).toMessage());
      return;
    }
    case MessageType::SCRIPT_REMOTE_CALL: {
      auto& scriptRemoteCall = static_cast<ScriptRemoteCall&>(rpc);
      auto rrefId = scriptRemoteCall.retRRefId();
      auto forkId = scriptRemoteCall.retForkId();
      auto& ctx = RRefContext::getInstance();

      TypePtr returnType;
      if (scriptRemoteCall.hasOp()) {
        returnType = scriptRemoteCall.op()->schema().returns()[0].type();
      } else {
        returnType = PythonRpcHandler::getInstance()
                         .jitCompilationUnit()
                         ->get_function(scriptRemoteCall.qualifiedName())
                         .getSchema()
                         .returns()
                         .at(0)
                         .type();
      }

      auto ownerRRef = ctx.getOrCreateOwnerRRef(rrefId, returnType);
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
      auto& stack = scriptRemoteCall.stackRef();
      if (scriptRemoteCall.hasOp()) {
        try {
          scriptRemoteCall.op()->getOperation()(stack);
        } catch (const std::exception& e) {
          // Don't throw in this call, but rather transfer the exception
          // to the rref.
          ownerRRef->setError(e.what());
          postProcessing();
          return;
        }
        TORCH_INTERNAL_ASSERT(
            stack.size() == 1,
            "Return value of a builtin operator or a "
            "TorchScript function should be a single IValue, got a vector of "
            "size ",
            stack.size());
        ownerRRef->setValue(std::move(stack.front()));
        postProcessing();
        return;
      }

      c10::intrusive_ptr<c10::ivalue::Future> jitFuture;
      try {
        jitFuture = PythonRpcHandler::getInstance()
                        .jitCompilationUnit()
                        ->get_function(scriptRemoteCall.qualifiedName())
                        .runAsync(stack);
        if (jitFuture->completed()) { // short-cut.
          ownerRRef->setValue(jitFuture->value());
          postProcessing();
          return;
        }
      } catch (const std::exception& e) {
        ownerRRef->setError(e.what());
        postProcessing();
        return;
      }
      jitFuture->addCallback([ownerRRef, postProcessing, jitFuture]() {
        try {
          ownerRRef->setValue(jitFuture->value());
        } catch (const std::exception& e) {
          ownerRRef->setError(e.what());
        }
        postProcessing();
      });
      return;
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      auto& uprc = static_cast<UnpickledPythonRemoteCall&>(rpc);

      const auto& rrefId = uprc.rrefId();
      const auto& forkId = uprc.forkId();
      auto& ctx = RRefContext::getInstance();

      auto ownerRRef = ctx.getOrCreateOwnerRRef(rrefId, PyObjectType::get());

      auto& pythonRpcHandler = PythonRpcHandler::getInstance();
      IValue py_ivalue;
      try {
        {
          py::gil_scoped_acquire acquire;
          py_ivalue = jit::toIValue(
              pythonRpcHandler.runPythonUdf(std::move(uprc).movePythonUdf()),
              PyObjectType::get());
        }
        ownerRRef->setValue(std::move(py_ivalue));
      } catch (py::error_already_set& e) {
        // py::error_already_set requires GIL to destruct, take special care.
        ownerRRef->setError(e.what());
        py::gil_scoped_acquire acquire;
        e.restore();
        PyErr_Clear();
      } catch (std::exception& e) {
        ownerRRef->setError(e.what());
      }

      if (rrefId != forkId) {
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
      markComplete(RemoteRet(rrefId, forkId).toMessage());
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
                responseFuture->setError(whenValueSet->error()->what());
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
      // Making this lambda mutable to allow move-capture it in callbacks
      auto serialize = [](IValue value) mutable -> SerializedPyObj {
        auto& pythonRpcHandler = PythonRpcHandler::getInstance();
        // Need this GIL to guard jit::toPyObj and destruct its returned
        // py::object
        py::gil_scoped_acquire acquire;
        return pythonRpcHandler.serialize(jit::toPyObject(std::move(value)));
      };
      auto& prf = static_cast<PythonRRefFetchCall&>(rpc);
      auto& ctx = RRefContext::getInstance();

      auto futureOwner = ctx.getOwnerRRef(prf.rrefId());

      if (futureOwner->completed()) { // optional fast-path
        // the OwnerRRef has been created
        const auto& rref = futureOwner->constValue();
        if (rref->hasValue()) {
          SerializedPyObj result = serialize(rref->getValue());
          markComplete(
              PythonRRefFetchRet(std::move(result).toIValues()).toMessage());
          return;
        }
      }

      futureOwner->addCallback([responseFuture,
                                messageId,
                                futureOwner,
                                serialize{std::move(serialize)}]() mutable {
        const auto& rref = futureOwner->constValue();
        auto whenValueSet = rref->getFuture();

        // Our response is satisfied when the the rpc.remote() request
        // finishes executing on the owner.
        whenValueSet->addCallback([responseFuture,
                                   messageId,
                                   rref,
                                   whenValueSet,
                                   serialize{std::move(serialize)}]() mutable {
          if (whenValueSet->hasError()) {
            responseFuture->setError(whenValueSet->error()->what());
            return;
          }
          try {
            SerializedPyObj result = serialize(rref->getValue());
            Message m =
                PythonRRefFetchRet(std::move(result).toIValues()).toMessage();
            m.setId(messageId);
            responseFuture->markCompleted(std::move(m));
          } catch (const std::exception& e) {
            responseFuture->setError(e.what());
          }
        });
      });

      return;
    }
    case MessageType::RREF_USER_DELETE: {
      auto& rud = static_cast<RRefUserDelete&>(rpc);
      auto& ctx = RRefContext::getInstance();
      auto deletedRRef = ctx.delForkOfOwner(rud.rrefId(), rud.forkId());
      if (deletedRRef && deletedRRef->isPyObj()) {
        py::gil_scoped_acquire acquire;
        deletedRRef.reset();
      }
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
      // Kick off processing for the nested future and get a Future<T> to the
      // result.
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
            DistAutogradContextGuard ctxGuard(ctxId);

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
      execFuture->addCallback(
          [responseFuture, messageId](const FutureMessage& execFuture) {
            if (!execFuture.hasError()) {
              Message m = std::move(PropagateGradientsResp()).toMessage();
              m.setId(messageId);
              responseFuture->markCompleted(std::move(m));
            } else {
              responseFuture->setError(*(execFuture.error()));
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
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Request type ", messageType, " not supported.");
    }
  }
}

std::shared_ptr<FutureMessage> RequestCallbackImpl::processMessage(
    Message& request) const {
  // We need two futures here because it could pause twice when processing a
  // RPC message:
  //  1) waiting for all RRefs in the arguments to become confirmed;
  //  2) waiting for processRpc to finish.
  auto retFuture = std::make_shared<FutureMessage>();
  auto& rrefContext = RRefContext::getInstance();
  try {
    rrefContext.recordThreadLocalPendingRRefs();
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
          try {
            processRpc(*rpc, messageType, id, retFuture);
          } catch (py::error_already_set& e) {
            retFuture->markCompleted(handleError(e, messageType, id));
            // There are request callback impls in Python, where Python
            // exceptions could be thrown. For releasing Python exception
            // py::objects, GIL must be held.
            py::gil_scoped_acquire acquire;
            e.restore(); // Release ownership on py::objects and also restore
                         // Python Error Indicator.
            PyErr_Clear(); // Clear the Python Error Indicator as we has
                           // recorded the exception in the response message.
          } catch (std::exception& e) {
            retFuture->markCompleted(handleError(e, messageType, id));
          }
        });
  } catch (std::exception& e) {
    retFuture->markCompleted(handleError(e, request.type(), request.id()));
    rrefContext.clearRecordedPendingRRefsOnError();
  }
  return retFuture;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
