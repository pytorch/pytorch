#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/jit/pickle.h>

namespace torch {
namespace distributed {
namespace rpc {

ScriptRemoteCall::ScriptRemoteCall(
    std::shared_ptr<Operator> op,
    std::vector<at::IValue>&& args,
    const RRefId& retRRefId,
    const ForkId& retForkId)
    : ScriptCall(std::move(op), std::move(args)),
      retRRefId_(retRRefId),
      retForkId_(retForkId) {}

Message ScriptRemoteCall::toMessage() const {
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

ScriptRemoteCall ScriptRemoteCall::fromMessage(const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value =
      jit::unpickle(payload, payload_size, nullptr, &message.tensors());
  auto values = value.toTuple()->elements();

  // remove the last element from values and convert it back to an RRef
  auto retForkId = RRefId::fromIValue(values.back());
  values.pop_back();
  auto retRRefId = ForkId::fromIValue(values.back());
  values.pop_back();

  auto op = ScriptCall::fromIValues(values);
  return ScriptRemoteCall(op, std::move(values), retRRefId, retForkId);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
