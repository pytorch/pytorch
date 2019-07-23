#include <torch/csrc/distributed/rpc/functions.h>

namespace torch {
namespace distributed {
namespace rpc {

void processRequestBlocking(
    std::string from, Message request, RpcAgent& agent) {
  switch (request.type()) {
    case MessageType::BUILTIN_OP: {
      BuiltinOp op = BuiltinOp::fromMessage(std::move(request));
      op.op()->getOperation()(op.stack());
      auto response = BuiltinRet(op.stack()).toMessage();
      response.setId(request.id());
      agent.send(from, std::move(response));
      break;
    }
    default: {
      throw std::runtime_error("Request type not supported.");
    }
  }
}

}
}
}
