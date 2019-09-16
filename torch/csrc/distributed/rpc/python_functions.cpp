#include <torch/csrc/distributed/rpc/python_functions.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

std::shared_ptr<Operator> matchBuiltinOp(
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs,
    Stack& stack) {
  Symbol symbol = Symbol::fromQualString(opName);
  if (symbol.is_aten()) {
    for (const auto& op : torch::jit::getAllOperatorsFor(symbol)) {
      try {
        // FIXME: This is temporary solution. We should at least refactor
        // ``createStackForSchema`` to avoid throwing an error.
        stack = torch::jit::createStackForSchema(
            op->schema(), args, kwargs, c10::nullopt);

      } catch (std::runtime_error& e) {
        VLOG(1) << "Couldn't match schema: " << op->schema()
                << " to args: " << args << " and kwargs: " << kwargs
                << ", reason: " << e.what();
        continue;
      }

      // Found the right op!
      return op;
    }
  }

  AT_ERROR(
      "Failed to match operator name ",
      opName,
      " and arguments "
      "(args: ",
      args,
      ", kwargs: ",
      kwargs,
      ") to a builtin operator");

  // builtin operators.
}

} // namespace

py::object toPyObj(const Message& message) {
  switch (message.type()) {
    case MessageType::SCRIPT_RET: {
      ScriptRet ret = ScriptRet::fromMessage(message);
      Stack stack;
      stack.push_back(ret.value());
      return torch::jit::createPyObjectForStack(std::move(stack));
    }
    case MessageType::PYTHON_RET: {
      return PythonRpcHandler::getInstance().loadPythonUDFResult(message);
    }
    case MessageType::EXCEPTION: {
      std::string err(message.payload().begin(), message.payload().end());
      throw std::runtime_error(err);
    }
    default: {
      AT_ERROR("Unrecognized response message type ", message.type());
    }
  }
}

std::shared_ptr<FutureMessage> pyRpcBuiltin(
    RpcAgent& agent,
    const WorkerId& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs) {
  Stack stack;
  auto op = matchBuiltinOp(opName, args, kwargs, stack);
  return agent.send(dst, ScriptCall(op, std::move(stack)).toMessage());
}

std::shared_ptr<RRef> pyRemoteBuiltin(
    RpcAgent& agent,
    const WorkerId& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs) {
  Stack stack;
  auto op = matchBuiltinOp(opName, args, kwargs, stack);

  auto& ctx = RRefContext::getInstance();
  auto userRRef = ctx->createUserRRef(dst.id_);
  agent.send(
      dst,
      ScriptRemoteCall(
          op,
          std::move(stack),
          userRRef->id().toIValue(),
          userRRef->forkId().toIValue())
          .toMessage());
  return userRRef;
}

std::shared_ptr<FutureMessage> pyRpcPythonUdf(
    RpcAgent& agent,
    const WorkerId& dst,
    const std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors) {
  std::vector<char> data(pickledPythonUDF.begin(), pickledPythonUDF.end());

  return agent.send(
      dst,
      Message(std::move(data), std::move(tensors), MessageType::PYTHON_CALL));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
