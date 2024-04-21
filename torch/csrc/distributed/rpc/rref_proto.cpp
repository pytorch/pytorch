#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <limits>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

c10::ivalue::TupleElements toIValues(const Message& message, MessageType type) {
  TORCH_INTERNAL_ASSERT(
      type == message.type(),
      "Expecting message of type ",
      type,
      ", but got ",
      message.type());
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  return std::move(*std::move(value).toTuple()).elements();
}

c10::intrusive_ptr<Message> fromIValues(
    std::vector<IValue> ivalues,
    MessageType type) {
  std::vector<torch::Tensor> tensor_table;
  auto payload = jit::pickle(
      c10::ivalue::Tuple::create(std::move(ivalues)), &tensor_table);
  return c10::make_intrusive<Message>(
      std::move(payload), std::move(tensor_table), type);
}

} // namespace

/////////////////////////// RRefMessageBase //////////////////////////////////

const RRefId& RRefMessageBase::rrefId() {
  return rrefId_;
}

/////////////////////////// ForkMessageBase //////////////////////////////////

const ForkId& ForkMessageBase::forkId() {
  return forkId_;
}

c10::intrusive_ptr<Message> ForkMessageBase::toMessageImpl() && {
  return fromIValues({rrefId_.toIValue(), forkId_.toIValue()}, type_);
}

std::pair<RRefId, ForkId> ForkMessageBase::fromMessage(
    const Message& message,
    MessageType type) {
  auto ivalues = toIValues(message, type);

  TORCH_INTERNAL_ASSERT(
      ivalues.size() == 2, "ForkMessageBase expects 2 IValue from message.");

  return std::make_pair(
      RRefId::fromIValue(ivalues[0]), ForkId::fromIValue(ivalues[1]));
}

/////////////////////////// RRef Protocol //////////////////////////////////

c10::intrusive_ptr<Message> ScriptRRefFetchCall::toMessageImpl() && {
  std::vector<at::IValue> ivalues;
  ivalues.reserve(2);
  ivalues.emplace_back(rrefId_.toIValue());
  ivalues.emplace_back(fromWorkerId_);
  return fromIValues(std::move(ivalues), MessageType::SCRIPT_RREF_FETCH_CALL);
}

std::unique_ptr<ScriptRRefFetchCall> ScriptRRefFetchCall::fromMessage(
    const Message& message) {
  auto values = toIValues(message, MessageType::SCRIPT_RREF_FETCH_CALL);
  TORCH_INTERNAL_ASSERT(
      values.size() == 2, "ScriptRRefFetchCall expects 2 IValues from message");
  auto id = values[1].toInt();
  TORCH_INTERNAL_ASSERT(
      id >= std::numeric_limits<worker_id_t>::min() &&
          id <= std::numeric_limits<worker_id_t>::max(),
      "ScriptRRefFetchCall fromWorkerId exceeds worker_id_t limit.")
  return std::make_unique<ScriptRRefFetchCall>(
      worker_id_t(id), RRefId::fromIValue(values[0]));
}

c10::intrusive_ptr<Message> PythonRRefFetchCall::toMessageImpl() && {
  std::vector<at::IValue> ivalues;
  ivalues.reserve(2);
  ivalues.emplace_back(rrefId_.toIValue());
  ivalues.emplace_back(fromWorkerId_);
  return fromIValues(std::move(ivalues), MessageType::PYTHON_RREF_FETCH_CALL);
}

std::unique_ptr<PythonRRefFetchCall> PythonRRefFetchCall::fromMessage(
    const Message& message) {
  auto values = toIValues(message, MessageType::PYTHON_RREF_FETCH_CALL);
  TORCH_INTERNAL_ASSERT(
      values.size() == 2, "PythonRRefFetchCall expects 2 IValues from message");
  auto id = values[1].toInt();
  TORCH_INTERNAL_ASSERT(
      id >= std::numeric_limits<worker_id_t>::min() &&
          id <= std::numeric_limits<worker_id_t>::max(),
      "PythonRRefFetchCall fromWorkerId exceeds worker_id_t limit.")
  return std::make_unique<PythonRRefFetchCall>(
      worker_id_t(id), RRefId::fromIValue(values[0]));
}

const std::vector<at::IValue>& RRefFetchRet::values() {
  return values_;
}

c10::intrusive_ptr<Message> RRefFetchRet::toMessageImpl() && {
  return fromIValues(values_, type_);
}

std::unique_ptr<ScriptRRefFetchRet> ScriptRRefFetchRet::fromMessage(
    const Message& message) {
  auto values = toIValues(message, MessageType::SCRIPT_RREF_FETCH_RET);
  TORCH_INTERNAL_ASSERT(
      values.size() == 1,
      "RRef of IValue should contain a single IValue, but got ",
      values.size());
  return std::make_unique<ScriptRRefFetchRet>(std::move(values).vec());
}

std::unique_ptr<PythonRRefFetchRet> PythonRRefFetchRet::fromMessage(
    const Message& message) {
  return std::make_unique<PythonRRefFetchRet>(
      toIValues(message, MessageType::PYTHON_RREF_FETCH_RET).vec());
}

std::unique_ptr<RRefUserDelete> RRefUserDelete::fromMessage(
    const Message& message) {
  auto pair =
      ForkMessageBase::fromMessage(message, MessageType::RREF_USER_DELETE);
  return std::make_unique<RRefUserDelete>(pair.first, pair.second);
}

std::unique_ptr<RemoteRet> RemoteRet::fromMessage(const Message& message) {
  auto pair = ForkMessageBase::fromMessage(message, MessageType::REMOTE_RET);
  return std::make_unique<RemoteRet>(pair.first, pair.second);
}

const ForkId& RRefChildAccept::forkId() const {
  return forkId_;
}

c10::intrusive_ptr<Message> RRefChildAccept::toMessageImpl() && {
  return fromIValues({forkId_.toIValue()}, MessageType::RREF_CHILD_ACCEPT);
}

std::unique_ptr<RRefChildAccept> RRefChildAccept::fromMessage(
    const Message& message) {
  auto values = toIValues(message, MessageType::RREF_CHILD_ACCEPT);
  TORCH_INTERNAL_ASSERT(values.size() == 1, "Expect 1 IValues from message.");

  return std::make_unique<RRefChildAccept>(ForkId::fromIValue(values.back()));
}

std::unique_ptr<RRefForkRequest> RRefForkRequest::fromMessage(
    const Message& message) {
  auto pair =
      ForkMessageBase::fromMessage(message, MessageType::RREF_FORK_REQUEST);
  return std::make_unique<RRefForkRequest>(pair.first, pair.second);
}

c10::intrusive_ptr<Message> RRefAck::toMessageImpl() && {
  return c10::make_intrusive<Message>(
      std::vector<char>{}, std::vector<torch::Tensor>{}, MessageType::RREF_ACK);
}

std::unique_ptr<RRefAck> RRefAck::fromMessage(const Message& message) {
  TORCH_INTERNAL_ASSERT(
      message.type() == MessageType::RREF_ACK,
      "Message type miss match, expect ",
      MessageType::RREF_ACK,
      ", but got ",
      message.type());
  return std::make_unique<RRefAck>();
}

} // namespace rpc
} // namespace distributed
} // namespace torch
