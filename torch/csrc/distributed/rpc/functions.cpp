#include <torch/csrc/distributed/rpc/functions.h>

namespace torch {
namespace distributed {
namespace rpc {

void processRequestBlocking(
    std::string from, Message request, RpcAgent& agent) {
  switch (request.type()) {
    case MessageType::BUILTIN_OP: {
      BuiltinOp op = BuiltinOp::fromMessage(request);
      op.op()->getOperation()(op.stack());
      auto ret = op.stack();
      auto response = BuiltinRet(std::move(ret)).toMessage();
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
