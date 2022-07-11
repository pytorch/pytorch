#include <ATen/ThreadLocalState.h>
#include <c10/util/C++17.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/python_call.h>
#include <torch/csrc/distributed/rpc/python_functions.h>
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
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/utils/python_compat.h>
#include <exception>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

IValue toPyIValue(const Message& message) {
  MessageType msgType = message.type();
  auto response = deserializeResponse(message, msgType);
  switch (msgType) {
    case MessageType::SCRIPT_RET: {
      auto& ret = static_cast<ScriptResp&>(*response);
      Stack stack;
      stack.push_back(ret.value());
      // Need GIL to guard createPyObjectForStack() and its returned
      // py::object
      py::gil_scoped_acquire acquire;
      return jit::toIValue(
          torch::jit::createPyObjectForStack(std::move(stack)),
          PyObjectType::get());
    }
    case MessageType::PYTHON_RET: {
      // TODO: Try to avoid a copy here.
      auto& resp = static_cast<PythonResp&>(*response);
      auto& pythonRpcHandler = PythonRpcHandler::getInstance();
      // Need GIL to destruct the py::object returned by deserialize()
      py::gil_scoped_acquire acquire;
      py::object value = pythonRpcHandler.deserialize(resp.serializedPyObj());
      pythonRpcHandler.handleException(value);
      return jit::toIValue(value, PyObjectType::get());
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

  std::shared_ptr<jit::Operator> matchedOperator;
  if (symbol.is_aten()) {
    // Prefer C10 ops so that they go through C10 dispatch. We expect the
    // total # of possible overloaded ops (i.e. size of below ops list) to be
    // small (i.e. it is 10 for torch.add) so a worst-case linear search should
    // not incur significant extra overhead.
    auto ops = torch::jit::getAllOperatorsFor(symbol);
    std::vector<std::shared_ptr<torch::jit::Operator>> c10OpsForSymbol;
    for (auto it = ops.begin(); it != ops.end();) {
      std::shared_ptr<jit::Operator> op = *it;
      if (op->isC10Op()) {
        c10OpsForSymbol.emplace_back(std::move(op));
        it = ops.erase(it);
      } else {
        ++it;
      }
    }

    // Don't throw on failures in this call, since we are not examining on all
    // operators here, and the matched operator may indeed not be a c10 op.
    std::pair<std::shared_ptr<torch::jit::Operator>, torch::jit::Stack>
        opWithStack;
    try {
      opWithStack = torch::jit::getOpWithStack(c10OpsForSymbol, args, kwargs);
    } catch (const std::runtime_error& e) {
      opWithStack = torch::jit::getOpWithStack(ops, args, kwargs);
    }
    matchedOperator = std::get<0>(opWithStack);
    stack = std::get<1>(opWithStack);
  }

  // We should never hit this path, since if !matchedOperator, then the last
  // call to getOpWithStack should have thrown.
  TORCH_CHECK(
      matchedOperator != nullptr,
      "Failed to match operator name ",
      opName,
      " and arguments "
      "(args: ",
      args,
      ", kwargs: ",
      kwargs,
      ") to a builtin operator");

  return matchedOperator;
}

c10::intrusive_ptr<JitFuture> sendPythonRemoteCall(
    const WorkerInfo& dst,
    SerializedPyObj serializedPyObj,
    const IValue& rrefId,
    const IValue& forkId,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution) {
  auto pythonRemoteCall = std::make_unique<PythonRemoteCall>(
      std::move(serializedPyObj), rrefId, forkId, isAsyncExecution);

  // set forceGradRecording to true as even if the args does not contain any
  // tensor, the return value might still contain tensors.
  auto agent = RpcAgent::getCurrentRpcAgent();
  return torch::distributed::autograd::sendMessageWithAutograd(
      *agent,
      dst,
      std::move(*pythonRemoteCall).toMessage(),
      true /*forceGradRecording*/,
      rpcTimeoutSeconds);
}

} // namespace

using namespace torch::distributed::autograd;

c10::intrusive_ptr<JitFuture> toPyJitFuture(
    const c10::intrusive_ptr<JitFuture>& messageJitFuture,
    bool hasValue) {
  if (hasValue) {
    auto child = messageJitFuture->createInstance(PyObjectType::get());
    messageJitFuture->addCallback(
        at::wrapPropagateTLSState([child](JitFuture& future) {
          if (future.hasError()) {
            child->setError(future.exception_ptr());
          } else {
            const Message& message = *future.value().toCustomClass<Message>();

            // toPyIValue might throw and we need to record the appropriate
            // exception.
            IValue ivalue;
            try {
              ivalue = toPyIValue(message);
            } catch (py::error_already_set& e) {
              py::gil_scoped_acquire acquire;
              // FIXME: this is a temporary solution to add a special-case for
              // ValueError and TypeError, as those are already used in our
              // tests. We should have a more comprehensive coverage for other
              // types of exceptions as well.
              if (e.matches(PyExc_ValueError)) {
                child->setErrorIfNeeded(
                    std::make_exception_ptr(pybind11::value_error(e.what())));
              } else if (e.matches(PyExc_TypeError)) {
                child->setErrorIfNeeded(
                    std::make_exception_ptr(pybind11::type_error(e.what())));
              } else {
                // py::error_already_set requires GIL to destruct, take special
                // care.
                child->setErrorIfNeeded(
                    std::make_exception_ptr(std::runtime_error(e.what())));
              }
              e.restore();
              PyErr_Clear();
              return;
            } catch (std::exception& e) {
              child->setErrorIfNeeded(std::current_exception());
              return;
            }

            child->markCompleted(ivalue, future.storages());
          }
        }));
    return child;
  } else {
    return messageJitFuture->then(
        at::wrapPropagateTLSState([](JitFuture& future) {
          if (future.hasError()) {
            std::rethrow_exception(future.exception_ptr());
          } else {
            return IValue();
          }
        }),
        NoneType::get());
  }
}

c10::intrusive_ptr<JitFuture> pyRpcBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs,
    const float rpcTimeoutSeconds) {
  DCHECK(PyGILState_Check());
  Stack stack;
  auto op = matchBuiltinOp(opName, args, kwargs, stack);
  // Release GIL since args and kwargs processing is done.
  py::gil_scoped_release release;
  auto scriptCall = std::make_unique<ScriptCall>(op, std::move(stack));
  auto agent = RpcAgent::getCurrentRpcAgent();
  return toPyJitFuture(sendMessageWithAutograd(
      *agent,
      dst,
      std::move(*scriptCall).toMessage(),
      false,
      rpcTimeoutSeconds));
}

c10::intrusive_ptr<JitFuture> pyRpcPythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution) {
  DCHECK(!PyGILState_Check());
  auto serializedPyObj =
      SerializedPyObj(std::move(pickledPythonUDF), std::move(tensors));
  auto pythonCall = std::make_unique<PythonCall>(
      std::move(serializedPyObj), isAsyncExecution);

  auto agent = RpcAgent::getCurrentRpcAgent();
  return toPyJitFuture(sendMessageWithAutograd(
      *agent,
      dst,
      std::move(*pythonCall).toMessage(),
      true /*forceGradRecording*/,
      rpcTimeoutSeconds));
}

c10::intrusive_ptr<JitFuture> pyRpcTorchscript(
    const std::string& dstWorkerName,
    const std::string& qualifiedNameStr,
    const py::tuple& argsTuple,
    const py::dict& kwargsDict,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution) {
  // No need to catch exception here, if function can not be found,
  // exception will be thrown in get_function() call; if args do not match
  // with function schema, exception will be thrown in
  // createStackForSchema() call.
  DCHECK(!PyGILState_Check());
  const c10::QualifiedName qualifiedName(qualifiedNameStr);
  auto functionSchema = PythonRpcHandler::getInstance()
                            .jitCompilationUnit()
                            ->get_function(qualifiedName)
                            .getSchema();
  Stack stack;
  {
    // Acquire GIL for py::args and py::kwargs processing.
    py::gil_scoped_acquire acquire;
    stack = torch::jit::createStackForSchema(
        functionSchema,
        argsTuple.cast<py::args>(),
        kwargsDict.cast<py::kwargs>(),
        c10::nullopt);
  }
  DCHECK(!PyGILState_Check());
  c10::intrusive_ptr<c10::ivalue::Future> fut = rpcTorchscript(
      dstWorkerName,
      qualifiedName,
      functionSchema,
      stack,
      rpcTimeoutSeconds,
      isAsyncExecution);
  return fut;
}

PyRRef pyRemoteBuiltin(
    const WorkerInfo& dst,
    const std::string& opName,
    const float rpcTimeoutSeconds,
    const py::args& args,
    const py::kwargs& kwargs) {
  DCHECK(PyGILState_Check());
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

    auto jitFuture = sendMessageWithAutograd(
        *agent,
        dst,
        std::move(*scriptRemoteCall).toMessage(),
        /*forceGradRecord */ false,
        /* timeout */ rpcTimeoutSeconds);

    userRRef->registerOwnerCreationFuture(jitFuture);
    ctx.addPendingUser(userRRef->forkId(), userRRef);
    jitFuture->addCallback(at::wrapPropagateTLSState(
        [forkId{userRRef->forkId()}](JitFuture& future) {
          callback::confirmPendingUser(future, forkId);
        }));
    return PyRRef(userRRef);
  } else {
    auto ownerRRef = ctx.createOwnerRRef(returnType);
    // prevent this owner RRef being deleted due to other forks
    ctx.addSelfAsFork(ownerRRef);

    auto scriptRemoteCall = std::make_unique<ScriptRemoteCall>(
        op, std::move(stack), ownerRRef->rrefId(), ownerRRef->rrefId());
    auto jitFuture = sendMessageWithAutograd(
        *agent,
        dst,
        std::move(*scriptRemoteCall).toMessage(),
        /* forceGradRecord */ false,
        /* timeout */ rpcTimeoutSeconds);

    ownerRRef->registerOwnerCreationFuture(jitFuture);
    // Builtin operators does not return py::object, and hence does not require
    // GIL for destructing the potentially deleted OwerRRef.
    jitFuture->addCallback(at::wrapPropagateTLSState(
        [ownerRRefId = ownerRRef->rrefId()](JitFuture& future) {
          callback::finishCreatingOwnerRRef(future, ownerRRefId);
        }));
    return PyRRef(ownerRRef);
  }
}

PyRRef pyRemotePythonUdf(
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution) {
  DCHECK(!PyGILState_Check());
  auto& ctx = RRefContext::getInstance();
  auto serializedPyObj =
      SerializedPyObj(std::move(pickledPythonUDF), std::move(tensors));

  if (ctx.getWorkerId() != dst.id_) {
    auto userRRef = ctx.createUserRRef(dst.id_, PyObjectType::get());
    auto jitFuture = sendPythonRemoteCall(
        dst,
        std::move(serializedPyObj),
        userRRef->rrefId().toIValue(),
        userRRef->forkId().toIValue(),
        rpcTimeoutSeconds,
        isAsyncExecution);

    userRRef->registerOwnerCreationFuture(jitFuture);
    ctx.addPendingUser(userRRef->forkId(), userRRef);
    jitFuture->addCallback(at::wrapPropagateTLSState(
        [forkId{userRRef->forkId()}](JitFuture& future) {
          callback::confirmPendingUser(future, forkId);
        }));
    return PyRRef(userRRef);
  } else {
    // Sending remote message to self
    auto ownerRRef = ctx.createOwnerRRef(PyObjectType::get());
    // prevent this owner RRef being deleted due to other forks
    ctx.addSelfAsFork(ownerRRef);
    auto jitFuture = sendPythonRemoteCall(
        dst,
        std::move(serializedPyObj),
        ownerRRef->rrefId().toIValue(),
        ownerRRef->rrefId().toIValue(),
        rpcTimeoutSeconds,
        isAsyncExecution);

    ownerRRef->registerOwnerCreationFuture(jitFuture);
    jitFuture->addCallback(at::wrapPropagateTLSState(
        [ownerRRefId = ownerRRef->rrefId()](JitFuture& future) {
          auto deletedRRef =
              callback::finishCreatingOwnerRRef(future, ownerRRefId);
          if (deletedRRef && deletedRRef->isPyObj()) {
            py::gil_scoped_acquire ag;
            deletedRRef.reset();
          }
        }));
    return PyRRef(ownerRRef);
  }
}

PyRRef pyRemoteTorchscript(
    const std::string& dstWorkerName,
    const std::string& qualifiedNameStr,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution,
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
    py::gil_scoped_acquire ag;
    stack = torch::jit::createStackForSchema(
        functionSchema, args, kwargs, c10::nullopt);
  }
  DCHECK(!PyGILState_Check());
  auto rrefPtr = remoteTorchscript(
      dstWorkerName,
      qualifiedName,
      functionSchema,
      stack,
      rpcTimeoutSeconds,
      isAsyncExecution);
  return PyRRef(rrefPtr);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
