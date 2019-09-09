#include <torch/csrc/distributed/rpc/script_rref_proto.h>
#include <torch/csrc/jit/pickle.h>

#include <limits>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

std::vector<IValue> toIValues(const Message& message, MessageType type) {
  TORCH_INTERNAL_ASSERT(type == message.type(),
      "Expecting message of type ",
      type,
      ", but got ",
      message.type());
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value =
      jit::unpickle(payload, payload_size, nullptr, &message.tensors());
  return value.toTuple()->elements();
}

Message fromIValues(std::vector<IValue> ivalues, MessageType type) {
  std::vector<torch::Tensor> tensor_table;
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);
  return Message(std::move(payload), std::move(tensor_table), type);
}

}

const at::IValue& RRefMessageBase::value() {
  return value_;
}

at::IValue& RRefMessageBase::valueRef() {
  return value_;
}

Message RRefMessageBase::toMessage() const {
  std::vector<at::IValue> ivalues;
  ivalues.emplace_back(value_);
  std::vector<torch::Tensor> tensor_table;
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);

  return Message(std::move(payload), std::move(tensor_table), type_);
}

at::IValue RRefMessageBase::fromMessage(const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value =
      jit::unpickle(payload, payload_size, nullptr, &message.tensors());
  auto values = value.toTuple()->elements();

  AT_ASSERT(values.size() == 1, "Expect a single IValue from message.");
  return std::move(values.front());
}

ScriptRRefFetchCall ScriptRRefFetchCall::fromMessage(const Message& message) {
  return ScriptRRefFetchCall(RRefMessageBase::fromMessage(message));
}

PythonRRefFetchCall PythonRRefFetchCall::fromMessage(const Message& message) {
  return PythonRRefFetchCall(RRefMessageBase::fromMessage(message));
}

ScriptRRefFetchRet ScriptRRefFetchRet::fromMessage(const Message& message) {
  return ScriptRRefFetchRet(RRefMessageBase::fromMessage(message));
}

ScriptUserDelete ScriptUserDelete::fromMessage(const Message& message) {
  return ScriptUserDelete(RRefMessageBase::fromMessage(message));
}

Message ScriptUserAccept::toMessage() {
  return fromIValues(
      {
        IValue(owner_),
        rrefId_.toIValue(),
        forkId_.toIValue()
      },
      MessageType::RREF_USER_ACCEPT
  );
}

ScriptUserAccept ScriptUserAccept::fromMessage(const Message& message) {
  auto values = toIValues(message, MessageType::RREF_USER_ACCEPT);

  TORCH_INTERNAL_ASSERT(values.size() == 3,
      "ScriptUserAccept expects 3 IValue from message.");

  ForkId forkId = ForkId::fromIValue(std::move(values.back()));
  values.pop_back();
  RRefId rrefId = ForkId::fromIValue(std::move(values.back()));
  values.pop_back();
  auto owner = values.back().toInt();
  TORCH_INTERNAL_ASSERT(owner < std::numeric_limits<worker_id_t>::max(),
      "owner id out of range ", owner);
  return ScriptUserAccept(worker_id_t(owner), rrefId, forkId);
}


Message ScriptForkNotify::toMessage() const {
  return fromIValues(
      {
          IValue(owner_),
          rrefId_.toIValue(),
          forkId_.toIValue(),
          IValue(forkDst_)
      },
      MessageType::RREF_FORK_NOTIFY
  );
}

ScriptForkNotify ScriptForkNotify::fromMessage(const Message& message) {
  auto values = toIValues(message, MessageType::RREF_FORK_NOTIFY);

  AT_ASSERT(values.size() == 4, "Expect 4 IValues from message.");
  auto forkDst = values.back().toInt();
  AT_ASSERT(forkDst < std::numeric_limits<worker_id_t>::max(),
      "Fork destination worker id out of bound ",
      forkDst);
  values.pop_back();
  RRefId rrefId = RRefId::fromIValue(std::move(values.back()));
  values.pop_back();
  ForkId forkId = ForkId::fromIValue(std::move(values.back()));
  values.pop_back();
  auto owner = values.back().toInt();
  TORCH_INTERNAL_ASSERT(owner < std::numeric_limits<worker_id_t>::max(),
      "owner id out of range ", owner);

  return ScriptForkNotify(owner, rrefId, forkId, forkDst);
}

Message ScriptForkAccept::toMessage() {
  return fromIValues({forkId_.toIValue()}, MessageType::RREF_FORK_ACCEPT);
}

ScriptForkAccept ScriptForkAccept::fromMessage(const Message& message) {
  auto values = toIValues(message, MessageType::RREF_FORK_ACCEPT);
  AT_ASSERT(values.size() == 1, "Expect 1 IValues from message.");

  return ScriptForkAccept(ForkId::fromIValue(values.back()));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
