#include <torch/csrc/distributed/rpc/python_functions.h>

namespace torch {
namespace distributed {
namespace rpc {

py::object to_py_obj(const Message& message) {
  switch (message.type()) {
    case MessageType::SCRIPT_RET: {
      ScriptRet ret = ScriptRet::fromMessage(message);
      Stack stack;
      stack.push_back(ret.value());
      return torch::jit::createPyObjectForStack(std::move(stack));
    }
    case MessageType::PYTHON_RET: {
      return PythonRpcHandler::loadPythonUDFResult(message);
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

std::shared_ptr<FutureMessage> py_rpc_builtin(
    RpcAgent& agent,
    const WorkerId& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs) {
    // builtin operators.
    Symbol symbol = Symbol::fromQualString(opName);
    if (symbol.is_aten()) {
      Stack stack;
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

        // Found the right op! Send it along...
        return agent.send(dst, ScriptCall(op, std::move(stack)).toMessage());
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
}

std::shared_ptr<FutureMessage> py_rpc_python_udf(
    RpcAgent& agent,
    const WorkerId& dst,
    const std::string& pickledPythonUDF) {
  std::vector<char> data(pickledPythonUDF.begin(), pickledPythonUDF.end());
  std::vector<torch::Tensor> tensor_table;

  return agent.send(dst,
                    Message(std::move(data),
                            std::move(tensor_table),
                            MessageType::PYTHON_CALL));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
