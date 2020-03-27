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
#include <torch/csrc/distributed/rpc/torchscript_functions.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/python_compat.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

py::object toPyObjInternal(RpcCommandBase& rpc, MessageType messageType) {
  switch (messageType) {
    case MessageType::SCRIPT_RET: {
      auto& ret = static_cast<ScriptResp&>(rpc);
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
      auto& resp = static_cast<PythonResp&>(rpc);
      auto& pythonRpcHandler = PythonRpcHandler::getInstance();
      py::object ret = pythonRpcHandler.deserialize(resp.serializedPyObj());
      return ret;
    }
    default: {
      TORCH_CHECK(false, "Unrecognized response message type ", messageType);
    }
  }
}

py::object toPyObj(const Message& message) {
  MessageType msgType = message.type();
  auto response = deserializeResponse(message, msgType);
  return toPyObjInternal(*response, msgType);
}

c10::intrusive_ptr<c10::ivalue::Future> wrapFutureMessageInJitFuture(
    const std::shared_ptr<FutureMessage>& responseMessageFuture) {
  // Notice, even we can ask for the JIT type of the Python object,
  // there is no need to do that, because the return value of this utility
  // will be passed back to Python land eventually.

  // Create a JIT future and add it to FutureMessage's callback to set value
  // of the JIT future.
  auto ivalueFuturePtr =
      c10::make_intrusive<c10::ivalue::Future>(PyObjectType::get());
  responseMessageFuture->addCallback(
      [ivalueFuturePtr](
          const Message& responseMessage,
          const c10::optional<utils::FutureError>& futErr) {
        if (futErr) {
          c10::ivalue::Future::FutureError jitFutErr(
              std::string(futErr->what()));
          ivalueFuturePtr->markCompleted(std::move(jitFutErr));
        } else {
          ivalueFuturePtr->markCompleted(torch::jit::toIValue(
              toPyObj(responseMessage), PyObjectType::get()));
        }
      });
  return ivalueFuturePtr;
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
    const IValue& forkId,
    const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf) {
  auto pythonRemoteCall = std::make_unique<PythonRemoteCall>(
      std::move(serializedPyObj), rrefId, forkId);

  // set forceGradRecording to true as even if the args does not contain any
  // tensor, the return value might still contain tensors.
  auto agent = RpcAgent::getCurrentRpcAgent();
  return torch::distributed::autograd::sendMessageWithAutograd(
      *agent,
      dst,
      std::move(*pythonRemoteCall).toMessage(),
      true /*forceGradRecording*/,
      rf);
}

} // namespace

using namespace torch::distributed::autograd;

std::shared_ptr<jit::PythonFutureWrapper> pyRpcBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf,
    const py::args& args,
    const py::kwargs& kwargs) {
  DCHECK(PyGILState_Check());
  Stack stack;
  auto op = matchBuiltinOp(opName, args, kwargs, stack);
  // Release GIL since args and kwargs processing is done.
  py::gil_scoped_release release;
  auto scriptCall = std::make_unique<ScriptCall>(op, std::move(stack));
  auto agent = RpcAgent::getCurrentRpcAgent();
  auto responseMessageFuture = sendMessageWithAutograd(
      *agent, dst, std::move(*scriptCall).toMessage(), false, rf);

  // Notice, even we can get the JIT type of the Python object from the op
  // schema, there is no need to do that, because the return value will be
  // passed back to Python land eventually.
  return std::make_shared<torch::jit::PythonFutureWrapper>(
      wrapFutureMessageInJitFuture(responseMessageFuture));
}

std::shared_ptr<jit::PythonFutureWrapper> pyRpcPythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf) {
  DCHECK(!PyGILState_Check());
  auto serializedPyObj =
      SerializedPyObj(std::move(pickledPythonUDF), std::move(tensors));
  auto pythonCall = std::make_unique<PythonCall>(std::move(serializedPyObj));

  auto agent = RpcAgent::getCurrentRpcAgent();
  auto responseMessageFuture = sendMessageWithAutograd(
      *agent,
      dst,
      std::move(*pythonCall).toMessage(),
      true /*forceGradRecording*/,
      rf);

  return std::make_shared<torch::jit::PythonFutureWrapper>(
      wrapFutureMessageInJitFuture(responseMessageFuture),
      [](const py::object& value) {
        py::gil_scoped_release release;
        auto& pythonRpcHandler = PythonRpcHandler::getInstance();
        // This will unwrap RemoteException and raise the contained
        // server-side Python exception on client side. A caveat here is that
        // the exception must be raise in the client thread calling the pybind
        // "wait" API, so that it can be correctly shown to user. A wrong way is
        // to raise it in RPC server thread, where the exception would be
        // swallowed in the ThreadPool task, and also no pybind handling code
        // can help shown the Python exception.
        pythonRpcHandler.handleException(value);
      });
}

std::shared_ptr<jit::PythonFutureWrapper> pyRpcTorchscript(
    const std::string& dstWorkerName,
    const py::object& userCallable,
    const py::tuple& argsTuple,
    const py::dict& kwargsDict) {
  DCHECK(!PyGILState_Check());
  // No need to catch exception here, if function can not be found,
  // exception will be thrown in get_function() call; if args do not match
  // with function schema, exception will be thrown in
  // createStackForSchema() call.
  auto& pythonRpcHandler = PythonRpcHandler::getInstance();
  c10::QualifiedName qualifiedName =
      pythonRpcHandler.getQualifiedName(userCallable);
  c10::FunctionSchema functionSchema = pythonRpcHandler.jitCompilationUnit()
                                           ->get_function(qualifiedName)
                                           .getSchema();
  Stack stack;
  {
    py::gil_scoped_acquire acquire;
    stack = torch::jit::createStackForSchema(
        functionSchema,
        argsTuple.cast<py::args>(),
        kwargsDict.cast<py::kwargs>(),
        c10::nullopt);
  }
  DCHECK(!PyGILState_Check());
  c10::intrusive_ptr<c10::ivalue::Future> fut =
      rpcTorchscript(dstWorkerName, qualifiedName, functionSchema, stack);
  return std::make_shared<jit::PythonFutureWrapper>(fut);
}

PyRRef pyRemoteBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf,
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
        *agent, dst, std::move(*scriptRemoteCall).toMessage(), false, rf);

    ctx.addPendingUser(userRRef->forkId(), userRRef);
    fm->addCallback(callback::confirmPendingUser);
    return PyRRef(userRRef);
  } else {
    auto ownerRRef = ctx.createOwnerRRef(returnType);
    // prevent this owner RRef being deleted due to other forks
    ctx.addSelfAsFork(ownerRRef);

    auto scriptRemoteCall = std::make_unique<ScriptRemoteCall>(
        op, std::move(stack), ownerRRef->rrefId(), ownerRRef->rrefId());
    auto fm = sendMessageWithAutograd(
        *agent, dst, std::move(*scriptRemoteCall).toMessage(), false, rf);

    // Builtin operators does not return py::object, and hence does not require
    // GIL for destructing the potentially deleted OwerRRef.
    fm->addCallback(callback::finishCreatingOwnerRRef);
    return PyRRef(ownerRRef);
  }
}

PyRRef pyRemotePythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf) {
  auto& ctx = RRefContext::getInstance();
  auto serializedPyObj =
      SerializedPyObj(std::move(pickledPythonUDF), std::move(tensors));
  if (ctx.getWorkerId() != dst.id_) {
    auto userRRef = ctx.createUserRRef(dst.id_, PyObjectType::get());
    ctx.addPendingUser(userRRef->forkId(), userRRef);
    auto fm = sendPythonRemoteCall(
        dst,
        std::move(serializedPyObj),
        userRRef->rrefId().toIValue(),
        userRRef->forkId().toIValue(),
        rf);

    fm->addCallback(callback::confirmPendingUser);
    return PyRRef(userRRef);
  } else {
    auto ownerRRef = ctx.createOwnerRRef(PyObjectType::get());
    // prevent this owner RRef being deleted due to other forks
    ctx.addSelfAsFork(ownerRRef);
    auto fm = sendPythonRemoteCall(
        dst,
        std::move(serializedPyObj),
        ownerRRef->rrefId().toIValue(),
        ownerRRef->rrefId().toIValue(),
        rf);

    fm->addCallback([](const Message& message,
                       const c10::optional<utils::FutureError>& futErr) {
      auto deletedRRef = callback::finishCreatingOwnerRRef(message, futErr);
      if (deletedRRef && deletedRRef->isPyObj()) {
        pybind11::gil_scoped_acquire ag;
        deletedRRef.reset();
      }
    });
    return PyRRef(ownerRRef);
  }
}

PyRRef pyRemoteTorchscript(
    const std::string& dstWorkerName,
    const std::string& qualifiedNameStr,
    const py::args& args,
    const py::kwargs& kwargs) {
  DCHECK(!PyGILState_Check());
  auto qualifiedName = c10::QualifiedName(qualifiedNameStr);
  auto functionSchema = PythonRpcHandler::getInstance()
                            .jitCompilationUnit()
                            ->get_function(qualifiedName)
                            .getSchema();
  Stack stack;
  {
    // Acquire GIL for py::args and py::kwargs processing.
    pybind11::gil_scoped_acquire ag;
    stack = torch::jit::createStackForSchema(
        functionSchema, args, kwargs, c10::nullopt);
  }
  DCHECK(!PyGILState_Check());
  auto rrefPtr =
      remoteTorchscript(dstWorkerName, qualifiedName, functionSchema, stack);
  return PyRRef(rrefPtr);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
