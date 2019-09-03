#include <torch/csrc/distributed/rpc/script_ret.h>
#include <torch/csrc/jit/pickle.h>

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
  std::vector<char> payload;
  std::vector<jit::WriteableStorageData> storages;

  std::tie(payload, storages) = jit::pickle(value_);

  std::vector<torch::Tensor> tensor_table = fmap(
      storages,
      [](const jit::WriteableStorageData& data) { return data.tensor(); });

  return Message(std::move(payload),
                 std::move(tensor_table),
                 MessageType::SCRIPT_RET);
}

ScriptRet ScriptRet::fromMessage(const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();
  auto value = jit::unpickle(payload, payload_size, nullptr, &message.tensors());
  return ScriptRet(std::move(value));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
