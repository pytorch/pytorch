#include <torch/csrc/distributed/rpc/script_ret.h>
#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/unpickler.h>

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
  auto payload = jit::pickle(value_, &tensor_table);
  ;
  return Message(
      std::move(payload), std::move(tensor_table), MessageType::SCRIPT_RET);
}

ScriptRet ScriptRet::fromMessage(const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();
  auto value =
      jit::unpickle(payload, payload_size, nullptr, &message.tensors());
  return ScriptRet(std::move(value));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
