#include <torch/csrc/distributed/rpc/request_callback_impl.h>
#include <c10/util/C++17.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_context.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/python_udf_call.h>
#include <torch/csrc/distributed/rpc/python_udf_resp.h>
#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/distributed/rpc/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

using namespace torch::distributed::autograd;

Message RequestCallbackImpl::processRpc(
    RpcCommandBase& rpc,
    MessageType messageType) const {
  // TODO: RpcCommandBase should have an abstract execute() method that we can
  // call here instead of having another switch statement here. Even better we
  // could have abstract classes RpcRequest and RpcResp which inherit from
  // RpcCommandBase and RpcRequest declares the abstract method execute() that
  // we can call here. RpcResponse could have an abstract method to convert it
  // to a python object.
  switch (messageType) {
    case MessageType::SCRIPT_CALL: {
      auto& scriptCall = static_cast<ScriptCall&>(rpc);

      // sc is only alive within this block, use reference to avoid copy
      auto& stack = scriptCall.stackRef();
      scriptCall.op()->getOperation()(stack);

      TORCH_INTERNAL_ASSERT(
          stack.size() == 1,
          "Return value of a builtin operator or a "
          "TorchScript function should be a single IValue, got a vector of "
          "size ",
          stack.size());

      return std::move(ScriptResp(std::move(stack.front()))).toMessage();
    }
    case MessageType::PYTHON_CALL: {
      auto& pyCall = static_cast<PythonUDFCall&>(rpc);
      std::vector<torch::Tensor> responseTensorTable;
      auto payload = PythonRpcHandler::getInstance().generatePythonUDFResult(
          pyCall.pickledPayload(), pyCall.tensors(), responseTensorTable);
      return std::move(PythonUDFResp(
                           std::move(payload), std::move(responseTensorTable)))
          .toMessage();
    }
    case MessageType::SCRIPT_REMOTE_CALL: {
      auto agent = RpcAgent::getDefaultRpcAgent();
      std::cout << agent->getWorkerInfo().id_ << " ++ in script remote call \n" << std::flush;
      auto& src = static_cast<ScriptRemoteCall&>(rpc);
      auto& ctx = RRefContext::getInstance();

      auto ownerRRef = ctx.getOrCreateOwnerRRef<IValue>(src.retRRefId());

      // TODO: make this asynchronous
      // src is only alive within this block, use reference to avoid copy
      auto& stack = src.stackRef();
      std::cout << agent->getWorkerInfo().id_ << " ++ in script remote call before op \n" << std::flush;

      src.op()->getOperation()(stack);
      std::cout << agent->getWorkerInfo().id_ << " ++ in script remote call after op \n" << std::flush;

      TORCH_INTERNAL_ASSERT(
          stack.size() == 1,
          "Return value of a builtin operator or a "
          "TorchScript function should be a single IValue, got a vector of "
          "size ",
          stack.size());

      std::cout << agent->getWorkerInfo().id_ << " ++ in script remote call after assert \n" << std::flush;


      ownerRRef->setValue(std::move(stack.front()));
      std::cout << agent->getWorkerInfo().id_ << " ++ in script remote call set value \n" << std::flush;

      ctx.addForkOfOwner(src.retRRefId(), src.retForkId());
      return RemoteRet(src.retRRefId(), src.retForkId()).toMessage();
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      auto& prc = static_cast<PythonRemoteCall&>(rpc);

      auto rrefId = RRefId::fromIValue(prc.retRRefId());
      auto forkId = ForkId::fromIValue(prc.retForkId());
      auto& ctx = RRefContext::getInstance();

      auto ownerRRef = ctx.getOrCreateOwnerRRef<py::object>(rrefId);

      ownerRRef->setValue(
          PythonRpcHandler::getInstance().runPythonUDF(prc.serializedPyObj()));
      ctx.addForkOfOwner(rrefId, forkId);
      return RemoteRet(rrefId, forkId).toMessage();
    }
    case MessageType::SCRIPT_RREF_FETCH_CALL: {
      auto agent = RpcAgent::getDefaultRpcAgent();
      std::cout << agent->getWorkerInfo().id_ << " ++ in script fetch all for \n" << std::flush;
      auto& srf = static_cast<ScriptRRefFetchCall&>(rpc);
      auto& ctx = RRefContext::getInstance();
      // TODO: make this asynchronous
      std::shared_ptr<OwnerRRef<IValue>> rref =
          ctx.getOrCreateOwnerRRef<IValue>(srf.rrefId());
      std::cout << agent->getWorkerInfo().id_ << " ++ before getting value of " << rref->rrefId() << std::endl << std::flush;
      auto ret = {rref->getValue()};
      std::cout << agent->getWorkerInfo().id_ << " ++ got value \n" << std::flush;
      return ScriptRRefFetchRet(ret).toMessage();
    }
    case MessageType::PYTHON_RREF_FETCH_CALL: {
      auto& prf = static_cast<PythonRRefFetchCall&>(rpc);
      auto& ctx = RRefContext::getInstance();
      // TODO: make this asynchronous
      std::shared_ptr<OwnerRRef<py::object>> rref =
          ctx.getOrCreateOwnerRRef<py::object>(prf.rrefId());
      SerializedPyObj result =
          PythonRpcHandler::getInstance().serialize(rref->getValue());
      return PythonRRefFetchRet(result.toIValues()).toMessage();
    }
    case MessageType::RREF_USER_DELETE: {
      auto& rud = static_cast<RRefUserDelete&>(rpc);
      auto& ctx = RRefContext::getInstance();
      ctx.delForkOfOwner(rud.rrefId(), rud.forkId());
      return std::move(RRefAck()).toMessage();
    }
    case MessageType::RREF_CHILD_ACCEPT: {
      auto& rca = static_cast<RRefChildAccept&>(rpc);
      auto& ctx = RRefContext::getInstance();
      ctx.delPendingChild(rca.forkId());
      return std::move(RRefAck()).toMessage();
    }
    case MessageType::RREF_FORK_REQUEST: {
      auto& rfr = static_cast<RRefForkRequest&>(rpc);
      auto& ctx = RRefContext::getInstance();
      ctx.addForkOfOwner(rfr.rrefId(), rfr.forkId());
      return RRefAck().toMessage();
    }
    case MessageType::FORWARD_AUTOGRAD_REQ: {
      auto& rpcWithAutograd = static_cast<RpcWithAutograd&>(rpc);

      // Attach 'recv' autograd function.
      DistAutogradContext* autogradContext = addRecvRpcBackward(
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
      auto wrappedRpcResponse =
          processRpc(rpcWithAutograd.wrappedRpc(), wrappedMessageType);

      return getMessageWithAutograd(
          rpcWithAutograd.fromWorkerId(),
          std::move(wrappedRpcResponse),
          MessageType::FORWARD_AUTOGRAD_RESP);
    }
    case MessageType::BACKWARD_AUTOGRAD_REQ: {
      auto& gradientsCall = static_cast<PropagateGradientsReq&>(rpc);
      const auto& autogradMetadata = gradientsCall.getAutogradMetadata();

      // Retrieve the appropriate autograd context.
      auto& autogradContext =
          DistAutogradContainer::getInstance().retrieveContext(
              autogradMetadata.autogradContextId);

      // Lookup the appropriate 'send' function to enqueue.
      std::shared_ptr<SendRpcBackward> sendFunction =
          autogradContext.retrieveSendFunction(
              autogradMetadata.autogradMessageId);

      // Attach the gradients to the send function.
      sendFunction->setGrads(gradientsCall.getGrads());

      // Now execute the autograd graph using the "distributed engine."
      DistEngine::getInstance().executeSendFunction(
          autogradContext, sendFunction);

      return std::move(PropagateGradientsResp()).toMessage();
    }
    case MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ: {
      auto& cleanupContextReq = static_cast<CleanupAutogradContextReq&>(rpc);
      auto cleanupContextId = cleanupContextReq.getContextId();
      // release the context if it still exists on this thread. We need to check
      // if it exists since it may have been deleted by an in-flight RPC.
      // This can create nested RPCs if there are other nodes that get notified
      // to clean up their context.
      DistAutogradContainer::getInstance().releaseContextIfPresent(
          cleanupContextId);
      return std::move(CleanupAutogradContextResp()).toMessage();
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Request type ", messageType, " not supported.");
    }
  }
}

Message RequestCallbackImpl::processMessage(Message& request) const {
  std::unique_ptr<RpcCommandBase> rpc = deserializeRequest(request);
  auto responseMessage = processRpc(*rpc, request.type());
  responseMessage.setId(request.id());
  return responseMessage;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
