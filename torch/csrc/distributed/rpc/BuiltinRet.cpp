#include <torch/csrc/distributed/rpc/BuiltinRet.h>

namespace torch {
namespace distributed {
namespace rpc {

BuiltinRet::BuiltinRet(std::vector<at::IValue>&& values) : values_(values) {}

BuiltinRet::~BuiltinRet() = default;

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

  auto meta = pickler.stack();
  return Message(std::move(meta),
                 std::move(tensor_table),
                 MessageType::BUILTIN_RET);
}

BuiltinRet BuiltinRet::fromMessage(const Message& message) {
  auto meta = static_cast<const void*>(message.meta().data());
  auto meta_size = message.meta().size();
  Unpickler unpickler(meta, meta_size, &message.tensors(), nullptr);

  return BuiltinRet(unpickler.parse_ivalue_list());
}

} // namespace rpc
} // namespace distributed
} // namespace torch
