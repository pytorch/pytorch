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
#include <torch/csrc/jit/pybind_utils.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {


void finishAcceptUserRRef(
    const rpc::Message& message,
    const c10::optional<utils::FutureError>& futErr) {
  RRefContext::handleException(futErr);
  auto rr = RemoteRet::fromMessage(message);
  auto& ctx = RRefContext::getInstance();
  ctx.delPendingUser(rr->forkId());
}

void finishCreatingOwnerRRef(
    const Message& message,
    const c10::optional<utils::FutureError>& futErr) {
  RRefContext::handleException(futErr);
  auto rr = RemoteRet::fromMessage(message);
  TORCH_INTERNAL_ASSERT(
      rr->rrefId() == rr->forkId(),
      "Expecting an OwnerRRef as RemoteRet but got a fork.");
  auto& ctx = RRefContext::getInstance();
  ctx.delForkOfOwner(rr->rrefId(), rr->rrefId());
}

std::shared_ptr<FutureMessage> sendPythonRemoteCall(
    RpcAgent& agent,
    const WorkerInfo& dst,
    SerializedPyObj serializedPyObj,
    const IValue& rrefId,
    const IValue& forkId,
    const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf) {
  auto pythonRemoteCall = std::make_unique<PythonRemoteCall>(
      std::move(serializedPyObj), rrefId, forkId);

  // set forceGradRecording to true as even if the args does not contain any
  // tensor, the return value might still contain tensors.
  return torch::distributed::autograd::sendMessageWithAutograd(
      agent,
      dst,
      std::move(*pythonRemoteCall).toMessage(),
      true /*forceGradRecording*/,
      rf);
}

} // namespace

using namespace torch::distributed::autograd;

std::shared_ptr<FutureMessage> pyRpcBuiltin(
    RpcAgent& agent,
    const WorkerInfo& dst,
    const std::string& opName,
    const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf,
    const py::args& args,
    const py::kwargs& kwargs) {
  Stack stack;
  auto op = matchBuiltinOp(opName, args, kwargs, stack);
  auto scriptCall = std::make_unique<ScriptCall>(op, std::move(stack));
  return sendMessageWithAutograd(
      agent, dst, std::move(*scriptCall).toMessage(), false, rf);
}

RRefPtr RemoteBuiltin(
    RpcAgent& agent,
    const WorkerInfo& dst,
    const std::shared_ptr<Operator>& op,
    const Stack& stack) {
  TypePtr ret_type = op->schema().returns()[0].type();

  auto& ctx = RRefContext::getInstance();
  // TODO: support creating RRefs on a local object.
  TORCH_INTERNAL_ASSERT(
      ctx.getWorkerId() != dst.id_,
      "Does not support creating RRef on self yet.");
  auto userRRef = ctx.createUserRRef(dst.id_, ret_type);

  auto scriptRemoteCall = std::make_unique<ScriptRemoteCall>(
      op, std::move(stack), userRRef->rrefId(), userRRef->forkId());

  auto fm = sendMessageWithAutograd(
      agent, dst, std::move(*scriptRemoteCall).toMessage(), false, rf);

  ctx.addPendingUser(userRRef->forkId(), userRRef);
  fm->addCallback(finishAcceptUserRRef);
  return userRRef;
}

std::shared_ptr<FutureMessage> pyRpcPythonUdf(
    RpcAgent& agent,
    const WorkerInfo& dst,
    std::string& pickledPythonUDF,
    std::vector<torch::Tensor>& tensors,
    const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf) {
  auto pythonCall = std::make_unique<PythonCall>(
      std::vector<char>(pickledPythonUDF.begin(), pickledPythonUDF.end()),
      tensors);
  return sendMessageWithAutograd(
      agent,
      dst,
      std::move(*pythonCall).toMessage(),
      true /*forceGradRecording*/,
      rf);
}

PyRRef pyRemotePythonUdf(
    RpcAgent& agent,
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
        agent,
        dst,
        std::move(serializedPyObj),
        userRRef->rrefId().toIValue(),
        userRRef->forkId().toIValue(),
        rf);

    fm->addCallback(finishAcceptUserRRef);
    return PyRRef(userRRef);
  } else {
    auto ownerRRef = ctx.createOwnerRRef(PyObjectType::get());
    // prevent this owner RRef be deleted due to other forks
    ctx.addSelfAsFork(ownerRRef);
    auto fm = sendPythonRemoteCall(
        agent,
        dst,
        std::move(serializedPyObj),
        ownerRRef->rrefId().toIValue(),
        ownerRRef->rrefId().toIValue(),
        rf);

    fm->addCallback(finishCreatingOwnerRRef);
    return PyRRef(ownerRRef);
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
