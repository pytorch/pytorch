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

void processRequestBlocking(
    worker_id_t from, Message&& request, RpcAgent& agent) {
  switch (request.type()) {
    case MessageType::SCRIPT_CALL: {
      ScriptCall op = ScriptCall::fromMessage(request);

      auto stack = op.stack();
      op.op()->getOperation()(stack);
      AT_ASSERT(stack.size() == 1, "Return value of a builtin operator or a "
          "TorchScript function should be a single IValue, got a vector of "
          "size ", stack.size());

      auto response = ScriptRet(std::move(stack.front())).toMessage();
      response.setId(request.id());
      agent.send(from, std::move(response));
      break;
    }
    case MessageType::PYTHON_CALL: {
      std::vector<torch::Tensor> tensorTable;
      agent.send(
          from,
          Message(
              PythonRpcHandler::generatePythonUDFResult(request),
              std::move(tensorTable),
              MessageType::PYTHON_RET,
              request.id()));
      break;
    }
    case MessageType::REMOTE_CALL: {
      ScriptRemoteCall src = ScriptRemoteCall::fromMessage(request);

      auto stack = src.stack();
      src.op()->getOperation()(stack);
      AT_ASSERT(stack.size() == 1, "Return value of a builtin operator or a "
          "TorchScript function should be a single IValue, got a vector of "
          "size ", stack.size());

      auto rrefForkIValue = src.ret();
      RRefContext::getInstance()
          ->getOrCreateRRef<IValue>(std::move(rrefForkIValue))
          ->setValue(std::move(stack.front()));

      break;
    }
    case MessageType::RREF_FETCH: {
      ScriptRRefFetch srf = ScriptRRefFetch::fromMessage(request);
      // TODO: make this asynchronous
      std::shared_ptr<RRef> rref =
          RRefContext::getInstance()->getOrCreateOwnerRRef<IValue>(
              RRefId::fromIValue(std::move(srf.value()))
          );
      auto response = ScriptRRefValue(rref->getValue()).toMessage();
      response.setId(request.id());
      agent.send(from, std::move(response));
      break;
    }
    case MessageType::RREF_ADD_FORK: {
      ScriptRRefAdd sra = ScriptRRefAdd::fromMessage(request);
      RRefContext::getInstance()->addFork(std::move(sra.value()));
      break;
    }
    case MessageType::RREF_DEL_FORK: {
      ScriptRRefDel srd = ScriptRRefDel::fromMessage(request);
      RRefContext::getInstance()->delFork(std::move(srd.value()));
      break;
    }
    default: {
      AT_ERROR("Request type ", request.type(), " not supported.");
    }
  }
}

}
}
}
