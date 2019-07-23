#include <torch/csrc/distributed/rpc/python_functions.h>

namespace torch {
namespace distributed {
namespace rpc {

py::object to_py_obj(Message message) {
  switch (message.type()) {
    case MessageType::BUILTIN_RET: {
      BuiltinRet ret = BuiltinRet::fromMessage(std::move(message));
      Stack stack = ret.values();
      return torch::jit::createPyObjectForStack(std::move(stack));
    }
    default: {
      throw std::runtime_error("Cannot convert message to Python object");
    }
  }
}

std::shared_ptr<FutureMessage> py_rpc(
    RpcAgent& agent,
    std::string dstName,
    std::string opName,
    py::args args,
    py::kwargs kwargs) {
  if (opName.rfind("aten", 0) == 0) {
    // builtin operators
    Symbol symbol = Symbol::fromQualString(opName);
    for (auto op: torch::jit::getAllOperatorsFor(symbol)) {
      try {
        Stack stack = torch::jit::createStackForSchema(
            op->schema(), args, kwargs, c10::nullopt);
        return agent.send(dstName, BuiltinOp(op, stack).toMessage());
      } catch (std::runtime_error) {}
    }
    throw std::runtime_error("unrecognized function " + opName);
  } else {
    throw std::runtime_error("unrecognized RPC function name");
  }
}

}
}
}
