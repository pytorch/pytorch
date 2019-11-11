#include <torch/csrc/distributed/rpc/python_functions.h>

#include <c10/util/C++17.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/python_call.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_resp.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/pybind_utils.h>

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
}

void finishAcceptUserRRef(
    const rpc::Message& message,
    bool hasError,
    const utils::FutureError& futErr) {
  RRefContext::handleException(hasError, futErr);
  auto rr = RemoteRet::fromMessage(message);
  auto& ctx = RRefContext::getInstance();
  ctx.delPendingUser(rr->forkId());
}

} // namespace

using namespace torch::distributed::autograd;

py::object toPyObjInternal(RpcCommandBase& rpc, MessageType messageType) {
  switch (messageType) {
    case MessageType::SCRIPT_RET: {
      auto& ret = static_cast<ScriptResp&>(rpc);
      Stack stack;
      stack.push_back(ret.value());
      {
        AutoGIL ag;
        // The createPyObjectForStack does not acquire GIL, but creating a new
        // py::object requires GIL.
        return torch::jit::createPyObjectForStack(std::move(stack));
      }
    }
    case MessageType::PYTHON_RET: {
      // TODO: Try to avoid a copy here.
      auto& resp = static_cast<PythonResp&>(rpc);

      return PythonRpcHandler::getInstance().loadPythonUDFResult(
          resp.pickledPayload(), resp.tensors());
    }
    case MessageType::FORWARD_AUTOGRAD_RESP: {
      auto& rpcWithAutograd = static_cast<RpcWithAutograd&>(rpc);

      // Attach 'recv' autograd function.
      addRecvRpcBackward(
          rpcWithAutograd.autogradMetadata(),
          rpcWithAutograd.tensors(),
          rpcWithAutograd.fromWorkerId());

      // Handle the original RPC.
      auto wrappedMessageType = rpcWithAutograd.wrappedMessageType();
      return toPyObjInternal(rpcWithAutograd.wrappedRpc(), wrappedMessageType);
    }
    default: {
      AT_ERROR("Unrecognized response message type ", messageType);
    }
  }
}

py::object toPyObj(const Message& message) {
  return toPyObjInternal(*deserializeResponse(message), message.type());
}

std::shared_ptr<torch::utils::Future<Message>> pyRpcBuiltin(
    RpcAgent& agent,
    const WorkerInfo& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs) {
  Stack stack;
  auto op = matchBuiltinOp(opName, args, kwargs, stack);
  auto scriptCall = c10::guts::make_unique<ScriptCall>(op, std::move(stack));
  return sendMessageWithAutograd(
      agent, dst, std::move(*scriptCall).toMessage());
}

PyRRef pyRemoteBuiltin(
    RpcAgent& agent,
    const WorkerInfo& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs) {
  Stack stack;
  auto op = matchBuiltinOp(opName, args, kwargs, stack);

  auto& ctx = RRefContext::getInstance();
  // TODO: support creating RRefs on a local object.
  TORCH_INTERNAL_ASSERT(
      ctx.getWorkerId() != dst.id_,
      "Does not support creating RRef on self yet.");
  auto userRRef = ctx.createUserRRef<IValue>(dst.id_);

  auto scriptRemoteCall = c10::guts::make_unique<ScriptRemoteCall>(
      op, std::move(stack), userRRef->rrefId(), userRRef->forkId());

  auto fm = sendMessageWithAutograd(
      agent, dst, std::move(*scriptRemoteCall).toMessage());

  ctx.addPendingUser(userRRef->forkId(), userRRef);
  fm->addCallback(finishAcceptUserRRef);
  return PyRRef(userRRef);
}

std::shared_ptr<torch::utils::Future<Message>> pyRpcPythonUdf(
    RpcAgent& agent,
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors) {
  auto pythonCall = c10::guts::make_unique<PythonCall>(
      std::vector<char>(pickledPythonUDF.begin(), pickledPythonUDF.end()),
      tensors);
  return sendMessageWithAutograd(
      agent, dst, std::move(*pythonCall).toMessage());
}

PyRRef pyRemotePythonUdf(
    RpcAgent& agent,
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors) {
  auto& ctx = RRefContext::getInstance();
  // TODO: support creating RRefs on a local object.
  TORCH_INTERNAL_ASSERT(
      ctx.getWorkerId() != dst.id_,
      "Does not support creating RRef on self yet.");
  auto userRRef = ctx.createUserRRef<py::object>(dst.id_);

  auto pythonRemoteCall = c10::guts::make_unique<PythonRemoteCall>(
      SerializedPyObj(std::move(pickledPythonUDF), std::move(tensors)),
      userRRef->rrefId().toIValue(),
      userRRef->forkId().toIValue());

  // set forceGradRecording to true as even if the args does not contain any
  // tensor, the return value might still contain tensors.
  auto fm = sendMessageWithAutograd(
      agent,
      dst,
      std::move(*pythonRemoteCall).toMessage(),
      true /*forceGradRecording*/
  );

  ctx.addPendingUser(userRRef->forkId(), userRRef);
  fm->addCallback(finishAcceptUserRRef);
  return PyRRef(userRRef);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
