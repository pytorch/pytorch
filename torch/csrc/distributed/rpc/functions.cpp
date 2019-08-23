#include <torch/csrc/distributed/rpc/functions.h>

namespace torch {
namespace distributed {
namespace rpc {

void sendException(
    const std::string& from,
    const Message& request,
    RpcAgent& agent,
    const std::exception& e) {
  const char* err = e.what();
  std::vector<char> payload(err, err + strlen(err));
  agent.send(
      from,
      Message(
          std::move(payload),
          std::vector<torch::Tensor>(),
          MessageType::EXCEPTION,
          request.id()));
}

void processRequestBlocking(
    const std::string& from,
    Message&& request,
    RpcAgent& agent) {
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
        agent.send(from, std::move(response));
      } catch (std::exception& e) {
        sendException(from, request, agent, e);
      }
      break;
    }
    case MessageType::PYTHON_CALL: {
      try {
        auto payload = PythonRpcHandler::generatePythonUDFResult(request);
        agent.send(
            from,
            Message(
                std::move(payload),
                std::vector<torch::Tensor>(),
                MessageType::PYTHON_RET,
                request.id()));
      } catch (std::exception& e) {
        sendException(from, request, agent, e);
      }
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
