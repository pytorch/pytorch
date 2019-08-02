#include <torch/csrc/distributed/rpc/functions.h>

namespace torch {
namespace distributed {
namespace rpc {

void processRequestBlocking(
    std::string from, Message&& request, RpcAgent& agent) {
  switch (request.type()) {
    case MessageType::SCRIPT_CALL: {
      ScriptCall op = ScriptCall::fromMessage(request);
      auto stack = op.stack();
      op.op()->getOperation()(stack);
      auto response = ScriptRet(std::move(stack)).toMessage();
      response.setId(request.id());
      agent.send(std::move(from), std::move(response));
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
