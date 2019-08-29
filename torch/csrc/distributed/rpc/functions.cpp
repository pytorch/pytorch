#include <torch/csrc/distributed/rpc/functions.h>

#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref.h>
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
        ScriptCall op = ScriptCall::fromMessage(request);

        auto stack = op.stack();
        op.op()->getOperation()(stack);
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
        auto payload = PythonRpcHandler::generatePythonUDFResult(request);
        return Message(
            std::move(payload),
            std::vector<torch::Tensor>(),
            MessageType::PYTHON_RET,
            request.id());
      } catch (std::exception& e) {
        return createException(request, e);
      }
      break;
    }
    case MessageType::REMOTE_CALL: {
      ScriptRemoteCall src = ScriptRemoteCall::fromMessage(request);

      auto stack = src.stack();
      src.op()->getOperation()(stack);
      AT_ASSERT(stack.size() == 1, "Return value of a builtin operator or a "
          "TorchScript function should be a single IValue, got a vector of "
          "size ", stack.size());

      RRefContext::getInstance()
          ->getOrCreateRRef<IValue>(src.ret())
          ->setValue(std::move(stack.front()));

      return Message();
    }
    case MessageType::RREF_FETCH: {
      ScriptRRefFetch srf = ScriptRRefFetch::fromMessage(request);
      // TODO: make this asynchronous
      std::shared_ptr<RRef> rref =
          RRefContext::getInstance()->getOrCreateOwnerRRef<IValue>(
              RRefId::fromIValue(srf.value())
          );
      auto response = ScriptRRefValue(rref->getValue()).toMessage();
      response.setId(request.id());
      return response;
    }
    case MessageType::RREF_ADD_FORK: {
      ScriptRRefAdd sra = ScriptRRefAdd::fromMessage(request);
      RRefContext::getInstance()->addFork(sra.value());
      return Message();
    }
    case MessageType::RREF_DEL_FORK: {
      ScriptRRefDel srd = ScriptRRefDel::fromMessage(request);
      RRefContext::getInstance()->delFork(srd.value());
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
