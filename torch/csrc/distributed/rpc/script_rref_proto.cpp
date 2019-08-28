#include <torch/csrc/distributed/rpc/script_rref_proto.h>
#include <torch/csrc/jit/pickle.h>


namespace torch {
namespace distributed {
namespace rpc {


at::IValue ScriptRRefBase::value() {
  return value_;
}

Message ScriptRRefBase::toMessage() const {
  std::vector<at::IValue> ivalues;
  ivalues.push_back(value_);
  std::vector<torch::Tensor> tensor_table;
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);

  return Message(std::move(payload),
                 std::move(tensor_table),
                 type_);
}

at::IValue ScriptRRefBase::fromMessage(const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value = jit::unpickle(
      payload, payload_size, nullptr, &message.tensors());
  auto values = value.toTuple()->elements();

  AT_ASSERT(values.size() == 1, "Expect a single IValue from message.");
  return std::move(values.front());
}

ScriptRRefFetch ScriptRRefFetch::fromMessage(const Message& message) {
  return ScriptRRefFetch(ScriptRRefBase::fromMessage(message));
}

ScriptRRefValue ScriptRRefValue::fromMessage(const Message& message) {
  return ScriptRRefValue(ScriptRRefBase::fromMessage(message));
}

ScriptRRefAdd ScriptRRefAdd::fromMessage(const Message& message) {
  return ScriptRRefAdd(ScriptRRefBase::fromMessage(message));
}

ScriptRRefDel ScriptRRefDel::fromMessage(const Message& message) {
  return ScriptRRefDel(ScriptRRefBase::fromMessage(message));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
