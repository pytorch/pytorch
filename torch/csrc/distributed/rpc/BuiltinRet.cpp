#include <torch/csrc/distributed/rpc/BuiltinRet.h>

namespace torch {
namespace distributed {
namespace rpc {

std::vector<at::IValue>& BuiltinRet::values() {
  return values_;
}

Message BuiltinRet::toMessage() {
  std::vector<torch::Tensor> tensor_table;
  Pickler pickler(&tensor_table);

  pickler.start();
  pickler.startTuple();
  for (auto& value: values_) {
    pickler.addIValue(value);
  }
  pickler.endTuple();
  pickler.finish();

  return Message(pickler.stack(),
                 std::move(tensor_table),
                 MessageType::BUILTIN_RET);
}

BuiltinRet BuiltinRet::fromMessage(Message message) {
  auto meta = static_cast<const void*>(message.meta().data());
  auto meta_size = message.meta().size();
  Unpickler unpickler(meta, meta_size, &message.tensors(), nullptr);

  return BuiltinRet(unpickler.parse_ivalue_list());
}

} // namespace rpc
} // namespace distributed
} // namespace torch
