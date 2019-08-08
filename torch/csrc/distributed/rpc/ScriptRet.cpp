#include <torch/csrc/distributed/rpc/ScriptRet.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

using torch::jit::Pickler;
using torch::jit::Unpickler;

} // namespace

ScriptRet::ScriptRet(at::IValue&& value) : value_(value) {}

const at::IValue& ScriptRet::value() {
  return value_;
}

Message ScriptRet::toMessage() {
  std::vector<torch::Tensor> tensor_table;
  Pickler pickler(&tensor_table);

  pickler.protocol();
  pickler.startTuple();
  pickler.pushIValue(value_);
  pickler.endTuple();
  pickler.stop();

  auto payload = pickler.stack();
  return Message(std::move(payload),
                 std::move(tensor_table),
                 MessageType::SCRIPT_RET);
}

ScriptRet ScriptRet::fromMessage(const Message& message) {
  auto payload = static_cast<const void*>(message.payload().data());
  auto payload_size = message.payload().size();
  Unpickler unpickler(payload, payload_size, &message.tensors(), nullptr);

  auto values = unpickler.parse_ivalue_list();
  AT_ASSERT(values.size() == 1, "Return value of a builtin operator or a "
      "TorchScript function should be a single IValue, got a vector of size ",
      values.size());
  return ScriptRet(std::move(values.front()));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
