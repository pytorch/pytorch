#include <torch/csrc/distributed/rpc/functions.h>

#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_ret.h>

namespace torch {
namespace distributed {
namespace rpc {

Message createException(const Message& request, const std::exception& e) {
  const char* err = e.what();
  std::vector<char> payload(err, err + strlen(err));
  return Message(
      std::move(payload),
      std::vector<torch::Tensor>(),
      MessageType::EXCEPTION,
      request.id());
}

Message processRequestBlocking(const WorkerInfo& from, Message&& request) {
  try {
    switch (request.type()) {
      case MessageType::SCRIPT_CALL: {
        ScriptCall sc = ScriptCall::fromMessage(request);

        // sc is only alive within this block, use reference to avoid copy
        auto& stack = sc.stackRef();
        sc.op()->getOperation()(stack);

        AT_ASSERT(
            stack.size() == 1,
            "Return value of a builtin operator or a "
            "TorchScript function should be a single IValue, got a vector of "
            "size ",
            stack.size());
        return ScriptRet(std::move(stack.front())).toMessage();
      }
      case MessageType::PYTHON_CALL: {
        auto payload =
            PythonRpcHandler::generatePythonUDFResult(request, from.id_);
        return Message(
            std::move(payload),
            std::vector<torch::Tensor>(),
            MessageType::PYTHON_RET);
      }
      case MessageType::SCRIPT_REMOTE_CALL: {
        ScriptRemoteCall src = ScriptRemoteCall::fromMessage(request);

        auto& ctx = RRefContext::getInstance();

        auto ownerRRef = ctx->getOrCreateOwnerRRef<IValue>(src.retRRefId());

        // TODO: make this asynchronous
        // src is only alive within this block, use reference to avoid copy
        auto& stack = src.stackRef();
        src.op()->getOperation()(stack);
        AT_ASSERT(
            stack.size() == 1,
            "Return value of a builtin operator or a "
            "TorchScript function should be a single IValue, got a vector of "
            "size ",
            stack.size());

        ownerRRef->setValue(std::move(stack.front()));

        ctx->addForkOfOwner(src.retRRefId(), src.retForkId());
        return RemoteRet(src.retRRefId(), src.retForkId()).toMessage();
      }
      case MessageType::PYTHON_REMOTE_CALL: {
        PythonRemoteCall prc = PythonRemoteCall::fromMessage(request);

        auto rrefId = RRefId::fromIValue(prc.retRRefId());
        auto forkId = ForkId::fromIValue(prc.retForkId());
        auto& ctx = RRefContext::getInstance();

        auto ownerRRef = ctx->getOrCreateOwnerRRef<py::object>(rrefId);
        ownerRRef->setValue(PythonRpcHandler::runPythonUDF(prc.udf()));

        ctx->addForkOfOwner(rrefId, forkId);
        return RemoteRet(rrefId, forkId).toMessage();
      }
      case MessageType::SCRIPT_RREF_FETCH_CALL: {
        ScriptRRefFetchCall srf = ScriptRRefFetchCall::fromMessage(request);
        // TODO: make this asynchronous
        std::shared_ptr<OwnerRRef<IValue>> rref =
            RRefContext::getInstance()->getOrCreateOwnerRRef<IValue>(
                srf.rrefId());
        return RRefFetchRet(rref->getValue()).toMessage();
      }
      case MessageType::PYTHON_RREF_FETCH_CALL: {
        PythonRRefFetchCall prf = PythonRRefFetchCall::fromMessage(request);
        // TODO: make this asynchronous
        std::shared_ptr<OwnerRRef<py::object>> rref =
            RRefContext::getInstance()->getOrCreateOwnerRRef<py::object>(
                prf.rrefId());
        return RRefFetchRet(
                   PythonRpcHandler::serialize(rref->getValue(), from.id_))
            .toMessage();
      }
      case MessageType::RREF_USER_ACCEPT: {
        RRefUserAccept rua = RRefUserAccept::fromMessage(request);
        auto& ctx = RRefContext::getInstance();
        ctx->finishUserRRef(rua.rrefId(), rua.forkId());
        return Message({}, {}, MessageType::ACK);
      }
      case MessageType::RREF_USER_DELETE: {
        RRefUserDelete rud = RRefUserDelete::fromMessage(request);
        RRefContext::getInstance()->delForkOfOwner(rud.rrefId(), rud.forkId());
        return Message({}, {}, MessageType::ACK);
      }
      case MessageType::RREF_FORK_NOTIFY: {
        RRefForkNotify rfn = RRefForkNotify::fromMessage(request);
        auto& ctx = RRefContext::getInstance();
        return ctx->acceptForkRequest(
            rfn.rrefId(), rfn.forkId(), rfn.forkDst());
      }
      default: {
        AT_ERROR("Request type ", request.type(), " not supported.");
      }
    }
  } catch (std::exception& e) {
    return createException(request, e);
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
