#include <torch/csrc/distributed/rpc/request_callback_impl.h>

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
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace distributed {
namespace rpc {

using namespace torch::distributed::autograd;

std::shared_ptr<FutureMessage> RequestCallbackImpl::processRpc(
    RpcCommandBase& rpc,
    MessageType messageType,
    int64_t messageId) const {
  auto wrap = [messageId](Message m) {
    m.setId(messageId);
    return std::make_shared<FutureMessage>(std::move(m));
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
      } else {
        PythonRpcHandler::getInstance()
            .jitCompilationUnit()
            ->get_function(scriptCall.qualifiedName())
            .run(stack);
      }

      TORCH_INTERNAL_ASSERT(
          stack.size() == 1,
          "Return value of a builtin operator or a "
          "TorchScript function should be a single IValue, got a vector of "
          "size ",
          stack.size());

      return wrap(std::move(ScriptResp(std::move(stack.front()))).toMessage());
    }
    case MessageType::PYTHON_CALL: {
      auto& pyCall = static_cast<PythonCall&>(rpc);
      auto& pythonRpcHandler = PythonRpcHandler::getInstance();
      std::shared_ptr<SerializedPyObj> serializedPyObj = nullptr;
      {
        pybind11::gil_scoped_acquire ag;
        auto pythonUdf = pythonRpcHandler.deserialize(pyCall.serializedPyObj());
        serializedPyObj =
            std::make_shared<SerializedPyObj>(pythonRpcHandler.serialize(
                pythonRpcHandler.runPythonUdf(std::move(pythonUdf))));
      }
      return wrap(
          std::move(PythonResp(std::move(*serializedPyObj))).toMessage());
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

      // TODO: make this asynchronous
      // scriptRemoteCall is only alive within this block, use reference to
      // avoid copy
      auto& stack = scriptRemoteCall.stackRef();
      if (scriptRemoteCall.hasOp()) {
        scriptRemoteCall.op()->getOperation()(stack);
      } else {
        PythonRpcHandler::getInstance()
            .jitCompilationUnit()
            ->get_function(scriptRemoteCall.qualifiedName())
            .run(stack);
      }

      TORCH_INTERNAL_ASSERT(
          stack.size() == 1,
          "Return value of a builtin operator or a "
          "TorchScript function should be a single IValue, got a vector of "
          "size ",
          stack.size());

      ownerRRef->setValue(std::move(stack.front()));
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
      return wrap(RemoteRet(rrefId, forkId).toMessage());
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      auto& prc = static_cast<PythonRemoteCall&>(rpc);

      auto rrefId = RRefId::fromIValue(prc.retRRefId());
      auto forkId = ForkId::fromIValue(prc.retForkId());
      auto& ctx = RRefContext::getInstance();

      auto ownerRRef = ctx.getOrCreateOwnerRRef(rrefId, PyObjectType::get());

      auto& pythonRpcHandler = PythonRpcHandler::getInstance();
      IValue py_ivalue;
      {
        pybind11::gil_scoped_acquire ag;
        auto pythonUdf = pythonRpcHandler.deserialize(prc.serializedPyObj());
        py_ivalue = jit::toIValue(
            PythonRpcHandler::getInstance().runPythonUdf(std::move(pythonUdf)),
            PyObjectType::get());
      }

      ownerRRef->setValue(std::move(py_ivalue));

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
      return wrap(RemoteRet(rrefId, forkId).toMessage());
    }
    case MessageType::SCRIPT_RREF_FETCH_CALL: {
      auto& srf = static_cast<ScriptRRefFetchCall&>(rpc);
      auto& ctx = RRefContext::getInstance();
      c10::intrusive_ptr<OwnerRRef> rref = ctx.getOwnerRRef(srf.rrefId());
      if (rref->hasValue()) { // optional fast-path
        return wrap(ScriptRRefFetchRet({rref->getValue()}).toMessage());
      }
      auto whenValueSet = rref->getFuture();
      auto responseFuture = std::make_shared<FutureMessage>();

      // Our response is satisfied when the rpcs come back.
      whenValueSet->addCallback(
          [responseFuture, messageId, rref](
              const rpc::Message& /* unused */,
              const c10::optional<utils::FutureError>& error) {
            if (!error) {
              Message m = ScriptRRefFetchRet({rref->getValue()}).toMessage();
              m.setId(messageId);
              responseFuture->markCompleted(std::move(m));
            } else {
              responseFuture->setError(error->what());
            }
          });
      return responseFuture;
    }
    case MessageType::PYTHON_RREF_FETCH_CALL: {
      auto& prf = static_cast<PythonRRefFetchCall&>(rpc);
      auto& ctx = RRefContext::getInstance();
      c10::intrusive_ptr<OwnerRRef> rref = ctx.getOwnerRRef(prf.rrefId());
      if (rref->hasValue()) { // optional fast-path
        auto value = rref->getValue();
        py::object pyValue;
        {
          pybind11::gil_scoped_acquire ag;
          pyValue = torch::jit::toPyObject(std::move(value));
        }
        SerializedPyObj result =
            PythonRpcHandler::getInstance().serialize(pyValue);
        return wrap(
            PythonRRefFetchRet(std::move(result).toIValues()).toMessage());
      }

      auto whenValueSet = rref->getFuture();
      auto responseFuture = std::make_shared<FutureMessage>();

      // Our response is satisfied when the rpcs come back.
      whenValueSet->addCallback(
          [responseFuture, messageId, rref](
              const rpc::Message& /* unused */,
              const c10::optional<utils::FutureError>& error) {
            if (!error) {
              auto value = rref->getValue();
              py::object pyValue;
              {
                pybind11::gil_scoped_acquire ag;
                pyValue = torch::jit::toPyObject(std::move(value));
              }
              SerializedPyObj result =
                  PythonRpcHandler::getInstance().serialize(pyValue);
              Message m =
                  PythonRRefFetchRet(std::move(result).toIValues()).toMessage();
              m.setId(messageId);
              responseFuture->markCompleted(std::move(m));
            } else {
              responseFuture->setError(error->what());
            }
          });
      return responseFuture;
    }
    case MessageType::RREF_USER_DELETE: {
      auto& rud = static_cast<RRefUserDelete&>(rpc);
      auto& ctx = RRefContext::getInstance();
      auto deletedRRef = ctx.delForkOfOwner(rud.rrefId(), rud.forkId());
      if (deletedRRef && deletedRRef->isPyObj()) {
        pybind11::gil_scoped_acquire ag;
        deletedRRef.reset();
      }
      return wrap(std::move(RRefAck()).toMessage());
    }
    case MessageType::RREF_CHILD_ACCEPT: {
      auto& rca = static_cast<RRefChildAccept&>(rpc);
      auto& ctx = RRefContext::getInstance();
      ctx.delPendingChild(rca.forkId());
      return wrap(std::move(RRefAck()).toMessage());
    }
    case MessageType::RREF_FORK_REQUEST: {
      auto& rfr = static_cast<RRefForkRequest&>(rpc);
      auto& ctx = RRefContext::getInstance();
      ctx.addForkOfOwner(rfr.rrefId(), rfr.forkId());
      return wrap(RRefAck().toMessage());
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
      autogradContainer.setCurrentContextId(autogradContext->contextId());

      // Process the original RPC.
      auto wrappedMessageType = rpcWithAutograd.wrappedMessageType();
      // Kick off processing for the nested future and get a Future<T> to the
      // result.
      auto wrappedRpcResponseFuture = processRpc(
          rpcWithAutograd.wrappedRpc(), wrappedMessageType, messageId);

      // Make an overall future for the wrapped response.
      auto responseFuture = std::make_shared<rpc::FutureMessage>();
      auto fromWorkerId = rpcWithAutograd.fromWorkerId();
      // The original future needs to be marked as completed when the wrapped
      // one completes, with the autograd context information wrapped.
      wrappedRpcResponseFuture->addCallback(
          [responseFuture, messageId, fromWorkerId, wrappedRpcResponseFuture](
              const Message& /* unused */,
              const c10::optional<utils::FutureError>& error) {
            if (error) {
              // Propagate error to responseFuture if we had one.
              responseFuture->setError(error->what());
            } else {
              auto msg = getMessageWithAutograd(
                  fromWorkerId,
                  std::move(*wrappedRpcResponseFuture).moveValue(),
                  MessageType::FORWARD_AUTOGRAD_RESP);
              msg.setId(messageId);
              responseFuture->markCompleted(std::move(msg));
            }
          });
      return responseFuture;
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

      auto responseFuture = std::make_shared<rpc::FutureMessage>();

      // Now execute the autograd graph using the "distributed engine."
      auto execFuture = DistEngine::getInstance().executeSendFunctionAsync(
          autogradContext, sendFunction, gradientsCall.retainGraph());

      // Our response is satisfied when the rpcs come back.
      execFuture->addCallback(
          [responseFuture, messageId](
              const Message& /* unused */,
              const c10::optional<utils::FutureError>& error) {
            if (!error) {
              Message m = std::move(PropagateGradientsResp()).toMessage();
              m.setId(messageId);
              responseFuture->markCompleted(std::move(m));
            } else {
              responseFuture->setError(error->what());
            }
          });
      return responseFuture;
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
      return wrap(std::move(CleanupAutogradContextResp()).toMessage());
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Request type ", messageType, " not supported.");
    }
  }
}

std::shared_ptr<FutureMessage> RequestCallbackImpl::processMessage(
    Message& request) const {
  std::unique_ptr<RpcCommandBase> rpc = deserializeRequest(request);
  return processRpc(*rpc, request.type(), request.id());
}

} // namespace rpc
} // namespace distributed
} // namespace torch
