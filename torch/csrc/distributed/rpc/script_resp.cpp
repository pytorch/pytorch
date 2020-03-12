#include <torch/csrc/distributed/rpc/script_resp.h>

#include <c10/util/C++17.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/unpickler.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

using torch::jit::Pickler;
using torch::jit::Unpickler;

} // namespace

ScriptResp::ScriptResp(at::IValue&& value) : value_(value) {}

const at::IValue& ScriptResp::value() {
  return value_;
}

Message ScriptResp::toMessage() && {
  isInRpcCall = true;
  std::vector<torch::Tensor> tensor_table;
  auto payload = jit::pickle(value_, &tensor_table);
  isInRpcCall = false;
  return Message(
      std::move(payload), std::move(tensor_table), MessageType::SCRIPT_RET);
}

std::unique_ptr<ScriptResp> ScriptResp::fromMessage(const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();
  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      &message.tensors());
  return std::make_unique<ScriptResp>(std::move(value));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
