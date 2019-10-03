#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/jit/pickle.h>

#include <limits>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

std::vector<IValue> toIValues(const Message& message, MessageType type) {
  TORCH_INTERNAL_ASSERT(
      type == message.type(),
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
  auto payload = jit::pickle(
      c10::ivalue::Tuple::create(std::move(ivalues)), &tensor_table);
  return Message(std::move(payload), std::move(tensor_table), type);
}

} // namespace

/////////////////////////// RRefMessageBase //////////////////////////////////

const RRefId& RRefMessageBase::rrefId() {
  return rrefId_;
}

Message RRefMessageBase::toMessage() && {
  return fromIValues({rrefId_.toIValue()}, type_);
}

at::IValue RRefMessageBase::fromMessage(
    const Message& message,
    MessageType type) {
  auto values = toIValues(message, type);

  TORCH_INTERNAL_ASSERT(
      values.size() == 1, "ScriptUserDelete expects 1 IValue from message.");
  return std::move(values.back());
}

/////////////////////////// ForkMessageBase //////////////////////////////////

const ForkId& ForkMessageBase::forkId() {
  return forkId_;
}

Message ForkMessageBase::toMessage() && {
  return fromIValues({rrefId_.toIValue(), forkId_.toIValue()}, type_);
}

std::pair<RRefId, ForkId> ForkMessageBase::fromMessage(
    const Message& message,
    MessageType type) {
  auto ivalues = toIValues(message, type);

  TORCH_INTERNAL_ASSERT(
      ivalues.size() == 2, "ScriptUserDelete expects 2 IValue from message.");

  return std::make_pair(
      RRefId::fromIValue(ivalues[0]), ForkId::fromIValue(ivalues[1]));
}

/////////////////////////// RRef Protocol //////////////////////////////////

std::unique_ptr<ScriptRRefFetchCall> ScriptRRefFetchCall::fromMessage(
    const Message& message) {
  return c10::guts::make_unique<ScriptRRefFetchCall>(
      RRefId::fromIValue(RRefMessageBase::fromMessage(
          message, MessageType::SCRIPT_RREF_FETCH_CALL)));
}

std::unique_ptr<PythonRRefFetchCall> PythonRRefFetchCall::fromMessage(
    const Message& message) {
  return c10::guts::make_unique<PythonRRefFetchCall>(
      RRefId::fromIValue(RRefMessageBase::fromMessage(
          message, MessageType::PYTHON_RREF_FETCH_CALL)));
}

const std::vector<at::IValue>& RRefFetchRet::values() {
  return values_;
}

Message RRefFetchRet::toMessage() && {
  std::vector<at::IValue> ivalues = values_;
  std::vector<torch::Tensor> tensor_table;
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);

  return Message(
      std::move(payload), std::move(tensor_table), MessageType::RREF_FETCH_RET);
}

std::unique_ptr<RRefFetchRet> RRefFetchRet::fromMessage(
    const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value =
      jit::unpickle(payload, payload_size, nullptr, &message.tensors());
  auto values = value.toTuple()->elements();

  return c10::guts::make_unique<RRefFetchRet>(std::move(values));
}

std::unique_ptr<RRefUserDelete> RRefUserDelete::fromMessage(
    const Message& message) {
  auto pair =
      ForkMessageBase::fromMessage(message, MessageType::RREF_USER_DELETE);
  return c10::guts::make_unique<RRefUserDelete>(
      RRefUserDelete(pair.first, pair.second));
}

std::unique_ptr<RemoteRet> RemoteRet::fromMessage(const Message& message) {
  auto pair = ForkMessageBase::fromMessage(message, MessageType::REMOTE_RET);
  return c10::guts::make_unique<RemoteRet>(pair.first, pair.second);
}

const ForkId& RRefChildAccept::forkId() const {
  return forkId_;
}

Message RRefChildAccept::toMessage() && {
  return fromIValues({forkId_.toIValue()}, MessageType::RREF_CHILD_ACCEPT);
}

std::unique_ptr<RRefChildAccept> RRefChildAccept::fromMessage(
    const Message& message) {
  auto values = toIValues(message, MessageType::RREF_CHILD_ACCEPT);
  TORCH_INTERNAL_ASSERT(values.size() == 1, "Expect 1 IValues from message.");

  return c10::guts::make_unique<RRefChildAccept>(
      ForkId::fromIValue(values.back()));
}

std::unique_ptr<RRefForkRequest> RRefForkRequest::fromMessage(
    const Message& message) {
  auto pair =
      ForkMessageBase::fromMessage(message, MessageType::RREF_FORK_REQUEST);
  return c10::guts::make_unique<RRefForkRequest>(pair.first, pair.second);
}

Message RRefAck::toMessage() && {
  return Message({}, {}, MessageType::RREF_ACK);
}

std::unique_ptr<RRefAck> RRefAck::fromMessage(const Message& message) {
  TORCH_INTERNAL_ASSERT(
      message.type() == MessageType::RREF_ACK,
      "Message type miss match, expect ",
      MessageType::RREF_ACK,
      ", but got ",
      message.type());
  return c10::guts::make_unique<RRefAck>();
}

} // namespace rpc
} // namespace distributed
} // namespace torch
