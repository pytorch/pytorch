#include <torch/csrc/distributed/rpc/script_rref_proto.h>
#include <torch/csrc/jit/pickle.h>

#include <limits>

namespace torch {
namespace distributed {
namespace rpc {


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

  return Message(std::move(payload),
                 std::move(tensor_table),
                 type_);
}

at::IValue RRefMessageBase::fromMessage(const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value = jit::unpickle(
      payload, payload_size, nullptr, &message.tensors());
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

ScriptUserAccept ScriptUserAccept::fromMessage(const Message& message) {
  return ScriptUserAccept(RRefMessageBase::fromMessage(message));
}

worker_id_t ScriptForkNotify::forkDst() const {
  return forkDst_;
}

Message ScriptForkNotify::toMessage() const {
  std::vector<at::IValue> ivalues;
  ivalues.emplace_back(value_);
  ivalues.emplace_back(forkDst_);
  std::vector<torch::Tensor> tensor_table;
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);

  return Message(std::move(payload),
                 std::move(tensor_table),
                 type_);
}

ScriptForkNotify ScriptForkNotify::fromMessage(const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value = jit::unpickle(
      payload, payload_size, nullptr, &message.tensors());
  auto values = value.toTuple()->elements();

  AT_ASSERT(values.size() == 2, "Expect 2 IValues from message.");
  auto forkDst = values[1].toInt();
  AT_ASSERT(forkDst < (int64_t)std::numeric_limits<worker_id_t>::max,
      "Fork destination worker id our of bound ", forkDst);
  return ScriptForkNotify(values[0], (worker_id_t) forkDst);
}

ScriptForkAccept ScriptForkAccept::fromMessage(const Message& message) {
  return ScriptForkAccept(RRefMessageBase::fromMessage(message));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
