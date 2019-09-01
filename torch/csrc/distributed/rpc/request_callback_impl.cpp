#include <torch/csrc/distributed/rpc/request_callback_impl.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_ret.h>

namespace torch {
namespace distributed {
namespace rpc {

Message RequestCallbackImpl::processMessage(Message&& request) {
  switch (request.type()) {
    case MessageType::SCRIPT_CALL: {
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
      break;
    }
    case MessageType::PYTHON_CALL: {
      auto payload = PythonRpcHandler::generatePythonUDFResult(request);
      return Message(
          std::move(payload),
          std::vector<torch::Tensor>(),
          MessageType::PYTHON_RET,
          request.id());
      break;
    }
    default: {
      AT_ERROR("Request type ", request.type(), " not supported.");
    }
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
