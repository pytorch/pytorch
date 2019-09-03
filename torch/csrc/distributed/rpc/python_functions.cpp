#include <torch/csrc/distributed/rpc/python_functions.h>

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_ret.h>

#include <torch/csrc/jit/pybind_utils.h>



namespace torch {
namespace distributed {
namespace rpc {

namespace {

std::shared_ptr<Operator> match_builtin_op(
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs,
    Stack& stack) {
  if (opName.rfind("aten", 0) == 0) {
    // builtin operators.
    Symbol symbol = Symbol::fromQualString(opName);
    for (const auto& op : torch::jit::getAllOperatorsFor(symbol)) {
      try {
        // FIXME: This is temporary solution. We should at least refactor
        // ``createStackForSchema`` to avoid throwing an error.
        stack = torch::jit::createStackForSchema(
            op->schema(), args, kwargs, c10::nullopt);

        return op;
      } catch (std::runtime_error) {}
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

} // namespace

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
  Stack stack;
  auto op = match_builtin_op(opName, args, kwargs, stack);
  return agent.send(dst, ScriptCall(op, std::move(stack)).toMessage());
}

PyRRef py_remote_builtin(
    RpcAgent& agent,
    const WorkerId& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs) {

  Stack stack;
  auto op = match_builtin_op(opName, args, kwargs, stack);

  auto& ctx = RRefContext::getInstance();
  if (ctx->getWorkerId() == dst.id_) {
    auto ownerRRef = ctx->createOwnerRRef<IValue>(dst.id_);
    agent.send(
        dst,
        ScriptRemoteCall(
            op,
            std::move(stack),
            ownerRRef->id().toIValue(),
            ownerRRef->id().toIValue()
        ).toMessage()
    );
    return PyRRef(ownerRRef);
  } else {
    auto userRRef = ctx->createUserRRef<IValue>(dst.id_);
    agent.send(
        dst,
        ScriptRemoteCall(
            op,
            std::move(stack),
            userRRef->id().toIValue(),
            userRRef->forkId().toIValue()
        ).toMessage()
    );
    return PyRRef(userRRef);
  }
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

PyRRef py_remote_python_udf(
    RpcAgent& agent,
    const WorkerId& dst,
    const std::string& pickledPythonUDF) {

  auto& ctx = RRefContext::getInstance();
  if (ctx->getWorkerId() == dst.id_) {
    auto ownerRRef = ctx->createOwnerRRef<py::object>(dst.id_);
    agent.send(
        dst,
        PythonRemoteCall(
            pickledPythonUDF,
            ownerRRef->id().toIValue(),
            ownerRRef->id().toIValue()
        ).toMessage()
    );
    return PyRRef(ownerRRef);
  } else {
    auto userRRef = ctx->createUserRRef<py::object>(dst.id_);
    agent.send(
        dst,
        PythonRemoteCall(
            pickledPythonUDF,
            userRRef->id().toIValue(),
            userRRef->forkId().toIValue()
        ).toMessage()
    );
    return PyRRef(userRRef);
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
