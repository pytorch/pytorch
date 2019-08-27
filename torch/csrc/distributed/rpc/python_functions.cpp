#include <torch/csrc/distributed/rpc/python_functions.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <torch/csrc/distributed/autograd/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

using namespace torch::distributed::autograd;

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
    const std::string& dstName,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs) {
  if (opName.rfind("aten", 0) == 0) {
    // builtin operators.
    Symbol symbol = Symbol::fromQualString(opName);
    for (const auto& op : torch::jit::getAllOperatorsFor(symbol)) {
      try {
        // FIXME: This is temporary solution. We should at least refactor
        // ``createStackForSchema`` to avoid throwing an error.
        Stack stack = torch::jit::createStackForSchema(
            op->schema(), args, kwargs, c10::nullopt);

        auto& autogradContainer = DistAutogradContainer::getInstance();
        if (autogradContainer.hasValidContext()) {
          // Get all the ivalues for the operator.
          auto ivalues =
              torch::jit::last(stack, op->schema().arguments().size());

          // Attach the appropriate autograd edges.
          auto grad_fn = addSendRpcBackward(ivalues);

          // Record the send function in our current context.
          auto& currentContext = autogradContainer.currentContext();
          currentContext.addSendFunction(grad_fn);
        }

        return agent.send(
            dstName, ScriptCall(op, std::move(stack)).toMessage());
      } catch (std::runtime_error) {
      }
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
    const std::string& dstName,
    const std::string& pickledPythonUDF) {
  std::vector<char> data(pickledPythonUDF.begin(), pickledPythonUDF.end());
  std::vector<torch::Tensor> tensor_table;

  return agent.send(
      dstName,
      Message(
          std::move(data), std::move(tensor_table), MessageType::PYTHON_CALL));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
