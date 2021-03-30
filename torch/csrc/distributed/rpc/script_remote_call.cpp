#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>

#include <c10/util/C++17.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch {
namespace distributed {
namespace rpc {

ScriptRemoteCall::ScriptRemoteCall(
    std::shared_ptr<Operator> op,
    std::vector<at::IValue>&& stack,
    const RRefId& retRRefId,
    const ForkId& retForkId)
    : ScriptCall(std::move(op), std::move(stack)),
      retRRefId_(retRRefId),
      retForkId_(retForkId) {}

ScriptRemoteCall::ScriptRemoteCall(
    const c10::QualifiedName& qualifiedName,
    std::vector<at::IValue>&& stack,
    const RRefId& retRRefId,
    const ForkId& retForkId,
    const bool isAsyncExecution)
    : ScriptCall(qualifiedName, std::move(stack), isAsyncExecution),
      retRRefId_(retRRefId),
      retForkId_(retForkId) {}

std::unique_ptr<ScriptRemoteCall> ScriptRemoteCall::fromIValues(
    std::vector<at::IValue>& ivalues) {
  // remove the last element from values and convert it back to an RRef
  auto retForkId = RRefId::fromIValue(ivalues.back());
  ivalues.pop_back();
  auto retRRefId = ForkId::fromIValue(ivalues.back());
  ivalues.pop_back();

  auto scriptCallPtr = ScriptCall::fromIValues(ivalues);

  if (scriptCallPtr->hasOp()) {
    return std::make_unique<ScriptRemoteCall>(
        scriptCallPtr->op(), std::move(ivalues), retRRefId, retForkId);
  } else {
    return std::make_unique<ScriptRemoteCall>(
        scriptCallPtr->qualifiedName(),
        std::move(ivalues),
        retRRefId,
        retForkId,
        scriptCallPtr->isAsyncExecution());
  }
}

Message ScriptRemoteCall::toMessageImpl() && {
  std::vector<IValue> ivalues;
  ScriptCall::toIValues(ivalues);
  ivalues.emplace_back(retRRefId_.toIValue());
  ivalues.emplace_back(retForkId_.toIValue());

  std::vector<torch::Tensor> tensor_table;
  auto payload = jit::pickle(
      c10::ivalue::Tuple::create(std::move(ivalues)), &tensor_table);

  return Message(
      std::move(payload),
      std::move(tensor_table),
      MessageType::SCRIPT_REMOTE_CALL);
}

std::unique_ptr<ScriptRemoteCall> ScriptRemoteCall::fromMessage(
    const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  auto values = value.toTuple()->elements();
  return fromIValues(values);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
