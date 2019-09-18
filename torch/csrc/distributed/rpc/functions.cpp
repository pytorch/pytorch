#include <torch/csrc/distributed/rpc/functions.h>

#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_ret.h>
#include <torch/csrc/distributed/rpc/script_rref_proto.h>

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

Message processRequestBlocking(Message&& request) {
  switch (request.type()) {
    case MessageType::SCRIPT_CALL: {
      try {
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
        auto response = ScriptRet(std::move(stack.front())).toMessage();

        response.setId(request.id());
        return response;
      } catch (std::exception& e) {
        return createException(request, e);
      }
      break;
    }
    case MessageType::PYTHON_CALL: {
      try {
        std::vector<torch::Tensor> tensorTable;
        auto payload = PythonRpcHandler::getInstance().generatePythonUDFResult(
            request, tensorTable);
        return Message(
            std::move(payload),
            std::move(tensorTable),
            MessageType::PYTHON_RET,
            request.id());
      } catch (std::exception& e) {
        return createException(request, e);
      }
      break;
    }
    case MessageType::REMOTE_CALL: {
      ScriptRemoteCall src = ScriptRemoteCall::fromMessage(request);

      auto rrefId = RRefId::fromIValue(src.retRRefId());
      auto forkId = ForkId::fromIValue(src.retForkId());
      TORCH_CHECK(rrefId != forkId, "Does not support remote call to self.");

      auto& ctx = RRefContext::getInstance();
      auto ownerRRef = ctx->getOrCreateOwnerRRef<IValue>(rrefId);

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
      return Message();
    }
    case MessageType::RREF_FETCH_CALL: {
      ScriptRRefFetchCall srf = ScriptRRefFetchCall::fromMessage(request);
      // TODO: make this asynchronous
      std::shared_ptr<OwnerRRef<IValue>> rref =
          RRefContext::getInstance()->getOrCreateOwnerRRef<IValue>(
              RRefId::fromIValue(srf.value()));
      auto response = ScriptRRefFetchRet(rref->getValue()).toMessage();
      response.setId(request.id());
      return response;
    }
    case MessageType::RREF_USER_CREATE: {
      ScriptRRefCreate sra = ScriptRRefCreate::fromMessage(request);
      RRefContext::getInstance()->addFork(sra.valueRef());
      return Message();
    }
    case MessageType::RREF_USER_DELETE: {
      ScriptRRefDelete srd = ScriptRRefDelete::fromMessage(request);
      RRefContext::getInstance()->delFork(srd.valueRef());
      return Message();
    }
    default: {
      AT_ERROR("Request type ", request.type(), " not supported.");
    }
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
