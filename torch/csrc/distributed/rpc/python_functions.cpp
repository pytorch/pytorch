#include <torch/csrc/distributed/rpc/python_functions.h>
#include <c10/util/C++17.h>
#include <torch/csrc/distributed/autograd/context/dist_autograd_container.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/python_udf_call.h>
#include <torch/csrc/distributed/rpc/python_udf_resp.h>
#include <torch/csrc/distributed/rpc/utils.h>

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>

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

void finishAcceptUserRRef(const Message& message) {
  RRefContext::handleException(message);
  auto rr = RemoteRet::fromMessage(message);
  auto& ctx = RRefContext::getInstance();
  ctx->delPendingUser(rr->forkId());
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
      auto& resp = static_cast<PythonUDFResp&>(rpc);

      return PythonRpcHandler::getInstance().loadPythonUDFResult(
          resp.pickledPayload(), resp.tensors());
    }
    case MessageType::MESSAGE_WITH_AUTOGRAD_RESP: {
      auto& rpcWithAutograd = static_cast<RpcWithAutograd&>(rpc);

      // Attach 'recv' autograd function.
      addRecvRpcBackward(
          rpcWithAutograd.autogradMetadata(), rpcWithAutograd.tensors());

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

std::shared_ptr<FutureMessage> pyRpcBuiltin(
    RpcAgent& agent,
    const WorkerInfo& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs) {
  Stack stack;
  auto op = matchBuiltinOp(opName, args, kwargs, stack);
  auto scriptCall = c10::guts::make_unique<ScriptCall>(op, std::move(stack));
  auto& autogradContainer = DistAutogradContainer::getInstance();
  if (autogradContainer.hasValidContext()) {
    // Retrieve the appropriate context to modify.
    auto& autogradContext = autogradContainer.currentContext();

    // Wrap the original rpc with autograd information.
    AutogradMetadata autogradMetadata(
        autogradContext.context_id(), autogradContainer.newAutogradMessageId());
    RpcWithAutograd rpcWithAutograd(
        MessageType::MESSAGE_WITH_AUTOGRAD_REQ,
        autogradMetadata,
        std::move(scriptCall));

    // Record autograd information for 'send'.
    addSendRpcBackward(
        autogradContext, autogradMetadata, rpcWithAutograd.tensors());

    return agent.send(dst, std::move(rpcWithAutograd).toMessage());
  } else {
    return agent.send(dst, std::move(*scriptCall).toMessage());
  }
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
      ctx->getWorkerId() != dst.id_,
      "Does not support creating RRef on self yet.");
  auto userRRef = ctx->createUserRRef<IValue>(dst.id_);
  auto fm = agent.send(
      dst,
      ScriptRemoteCall(
          op, std::move(stack), userRRef->rrefId(), userRRef->forkId())
          .toMessage());

  ctx->addPendingUser(userRRef->forkId(), userRRef);
  fm->addCallback(finishAcceptUserRRef);
  return PyRRef(userRRef);
}

std::shared_ptr<FutureMessage> pyRpcPythonUdf(
    RpcAgent& agent,
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors) {
  return agent.send(
      dst,
      PythonUDFCall(
          std::vector<char>(pickledPythonUDF.begin(), pickledPythonUDF.end()),
          tensors)
          .toMessage());
}

PyRRef pyRemotePythonUdf(
    RpcAgent& agent,
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors) {
  auto& ctx = RRefContext::getInstance();
  // TODO: support creating RRefs on a local object.
  TORCH_INTERNAL_ASSERT(
      ctx->getWorkerId() != dst.id_,
      "Does not support creating RRef on self yet.");
  auto userRRef = ctx->createUserRRef<py::object>(dst.id_);
  auto fm = agent.send(
      dst,
      PythonRemoteCall(
          SerializedPyObj(std::move(pickledPythonUDF), std::move(tensors)),
          userRRef->rrefId().toIValue(),
          userRRef->forkId().toIValue())
          .toMessage());

  ctx->addPendingUser(userRRef->forkId(), userRRef);
  fm->addCallback(finishAcceptUserRRef);
  return PyRRef(userRRef);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
