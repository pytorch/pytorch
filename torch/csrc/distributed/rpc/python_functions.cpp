#include <torch/csrc/distributed/rpc/python_functions.h>

#include <c10/util/C++17.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/python_call.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_resp.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

py::object toPyObj(const Message& message) {
  MessageType msgType = message.type();
  auto response = deserializeResponse(message, msgType);
  switch (msgType) {
    case MessageType::SCRIPT_RET: {
      auto& ret = static_cast<ScriptResp&>(*response);
      Stack stack;
      stack.push_back(ret.value());
      {
        pybind11::gil_scoped_acquire ag;
        // The createPyObjectForStack does not acquire GIL, but creating a new
        // py::object requires GIL.
        return torch::jit::createPyObjectForStack(std::move(stack));
      }
    }
    case MessageType::PYTHON_RET: {
      // TODO: Try to avoid a copy here.
      auto& resp = static_cast<PythonResp&>(*response);
      auto& pythonRpcHandler = PythonRpcHandler::getInstance();
      py::object ret = pythonRpcHandler.deserialize(resp.serializedPyObj());
      return ret;
    }
    default: {
      TORCH_CHECK(false, "Unrecognized response message type ", msgType);
    }
  }
}

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

  TORCH_CHECK(
      false,
      "Failed to match operator name ",
      opName,
      " and arguments "
      "(args: ",
      args,
      ", kwargs: ",
      kwargs,
      ") to a builtin operator");
}

std::shared_ptr<FutureMessage> sendPythonRemoteCall(
    const WorkerInfo& dst,
    SerializedPyObj serializedPyObj,
    const IValue& rrefId,
    const IValue& forkId) {
  auto pythonRemoteCall = std::make_unique<PythonRemoteCall>(
      std::move(serializedPyObj), rrefId, forkId);

  // set forceGradRecording to true as even if the args does not contain any
  // tensor, the return value might still contain tensors.
  auto agent = RpcAgent::getCurrentRpcAgent();
  return torch::distributed::autograd::sendMessageWithAutograd(
      *agent,
      dst,
      std::move(*pythonRemoteCall).toMessage(),
      true /*forceGradRecording*/);
}

void DeleteFutureIValue(FutureIValue* fv) {
  if (fv->constValue().isPyObject()) {
    pybind11::gil_scoped_acquire ag;
    delete fv;
  } else {
    delete fv;
  }
}

} // namespace

using namespace torch::distributed::autograd;

std::shared_ptr<FutureIValue> toFutureIValue(
    const std::shared_ptr<FutureMessage>& fm,
    bool hasValue) {
  if (hasValue) {
    // NB: The custom deleter is necessary because the FutureIValue object
    // holds a py::object and it would require GIL to delete.
    std::shared_ptr<FutureIValue> fv(new FutureIValue(), DeleteFutureIValue);

    fm->addCallback([fv](const FutureMessage& fm) {
      // Don't need to acquire GIL here, as toPyObj acquires GIL
      // when creating the py::object
      if (fm.hasError()) {
        fv->setError(*fm.error());
      } else {
        fv->markCompleted(
            jit::toIValue(toPyObj(fm.constValue()), PyObjectType::get()));
      }
    });

    return fv;
  } else {
    auto fv = std::make_shared<FutureIValue>();

    fm->addCallback([fv](const FutureMessage& fm) {
      if (fm.hasError()) {
        fv->setError(*fm.error());
      } else {
        fv->markCompleted(IValue());
      }
    });

    return fv;
  }
}

std::shared_ptr<FutureIValue> pyRpcBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs,
    const float rpcTimeoutSeconds) {
  Stack stack;
  auto op = matchBuiltinOp(opName, args, kwargs, stack);
  // Release GIL since args and kwargs processing is done.
  py::gil_scoped_release release;
  auto scriptCall = std::make_unique<ScriptCall>(op, std::move(stack));
  auto agent = RpcAgent::getCurrentRpcAgent();
  return toFutureIValue(sendMessageWithAutograd(
      *agent,
      dst,
      std::move(*scriptCall).toMessage(),
      false,
      rpcTimeoutSeconds));
}

PyRRef pyRemoteBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs) {
  Stack stack;
  auto op = matchBuiltinOp(opName, args, kwargs, stack);
  // Release GIL since args and kwargs processing is done.
  py::gil_scoped_release release;
  TypePtr returnType = op->schema().returns()[0].type();

  auto& ctx = RRefContext::getInstance();
  auto agent = RpcAgent::getCurrentRpcAgent();

  if (ctx.getWorkerId() != dst.id_) {
    auto userRRef = ctx.createUserRRef(dst.id_, returnType);

    auto scriptRemoteCall = std::make_unique<ScriptRemoteCall>(
        op, std::move(stack), userRRef->rrefId(), userRRef->forkId());

    auto fm = sendMessageWithAutograd(
        *agent, dst, std::move(*scriptRemoteCall).toMessage(), false);

    userRRef->registerOwnerCreationFuture(fm);
    ctx.addPendingUser(userRRef->forkId(), userRRef);
    fm->addCallback([forkId{userRRef->forkId()}](const FutureMessage& fm) {
      callback::confirmPendingUser(fm, forkId);
    });
    return PyRRef(userRRef);
  } else {
    auto ownerRRef = ctx.createOwnerRRef(returnType);
    // prevent this owner RRef being deleted due to other forks
    ctx.addSelfAsFork(ownerRRef);

    auto scriptRemoteCall = std::make_unique<ScriptRemoteCall>(
        op, std::move(stack), ownerRRef->rrefId(), ownerRRef->rrefId());
    auto fm = sendMessageWithAutograd(
        *agent, dst, std::move(*scriptRemoteCall).toMessage(), false);

    ownerRRef->registerOwnerCreationFuture(fm);

    // Builtin operators does not return py::object, and hence does not require
    // GIL for destructing the potentially deleted OwerRRef.
    fm->addCallback(
        [](const FutureMessage& fm) { callback::finishCreatingOwnerRRef(fm); });
    return PyRRef(ownerRRef);
  }
}

std::shared_ptr<FutureIValue> pyRpcPythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const float rpcTimeoutSeconds) {
  auto serializedPyObj =
      SerializedPyObj(std::move(pickledPythonUDF), std::move(tensors));
  auto pythonCall = std::make_unique<PythonCall>(std::move(serializedPyObj));

  auto agent = RpcAgent::getCurrentRpcAgent();
  return toFutureIValue(sendMessageWithAutograd(
      *agent,
      dst,
      std::move(*pythonCall).toMessage(),
      true /*forceGradRecording*/,
      rpcTimeoutSeconds));
}

PyRRef pyRemotePythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors) {
  auto& ctx = RRefContext::getInstance();
  auto serializedPyObj =
      SerializedPyObj(std::move(pickledPythonUDF), std::move(tensors));
  if (ctx.getWorkerId() != dst.id_) {
    auto userRRef = ctx.createUserRRef(dst.id_, PyObjectType::get());
    auto fm = sendPythonRemoteCall(
        dst,
        std::move(serializedPyObj),
        userRRef->rrefId().toIValue(),
        userRRef->forkId().toIValue());

    userRRef->registerOwnerCreationFuture(fm);

    ctx.addPendingUser(userRRef->forkId(), userRRef);
    fm->addCallback([forkId{userRRef->forkId()}](const FutureMessage& fm) {
      callback::confirmPendingUser(fm, forkId);
    });
    return PyRRef(userRRef);
  } else {
    auto ownerRRef = ctx.createOwnerRRef(PyObjectType::get());
    // prevent this owner RRef being deleted due to other forks
    ctx.addSelfAsFork(ownerRRef);
    auto fm = sendPythonRemoteCall(
        dst,
        std::move(serializedPyObj),
        ownerRRef->rrefId().toIValue(),
        ownerRRef->rrefId().toIValue());

    ownerRRef->registerOwnerCreationFuture(fm);

    fm->addCallback([](const FutureMessage& fm) {
      auto deletedRRef = callback::finishCreatingOwnerRRef(fm);
      if (deletedRRef && deletedRRef->isPyObj()) {
        pybind11::gil_scoped_acquire ag;
        deletedRRef.reset();
      }
    });
    return PyRRef(ownerRRef);
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
