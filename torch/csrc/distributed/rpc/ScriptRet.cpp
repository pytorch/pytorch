#include <torch/csrc/distributed/rpc/ScriptRet.h>

namespace torch {
namespace distributed {
namespace rpc {

ScriptRet::ScriptRet(std::vector<at::IValue>&& values) : values_(values) {}

const std::vector<at::IValue>& ScriptRet::values() {
  return values_;
}

Message ScriptRet::toMessage() {
  std::vector<torch::Tensor> tensor_table;
  Pickler pickler(&tensor_table);

  pickler.protocol();
  pickler.startTuple();
  for (auto& value: values_) {
    pickler.pushIValue(value);
  }
  pickler.endTuple();
  pickler.stop();

  auto meta = pickler.stack();
  return Message(std::move(meta),
                 std::move(tensor_table),
                 MessageType::SCRIPT_RET);
}

ScriptRet ScriptRet::fromMessage(const Message& message) {
  auto meta = static_cast<const void*>(message.meta().data());
  auto meta_size = message.meta().size();
  Unpickler unpickler(meta, meta_size, &message.tensors(), nullptr);

  return ScriptRet(unpickler.parse_ivalue_list());
}

} // namespace rpc
} // namespace distributed
} // namespace torch
