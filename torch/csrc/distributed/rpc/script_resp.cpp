#include <torch/csrc/distributed/rpc/script_resp.h>

#include <c10/util/C++17.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/unpickler.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

using torch::jit::Pickler;
using torch::jit::Unpickler;

} // namespace

ScriptResp::ScriptResp(at::IValue&& value, DeviceMap deviceMap) : value_(value), deviceMap_(std::move(deviceMap)) {}

const at::IValue& ScriptResp::value() {
  return value_;
}

c10::intrusive_ptr<Message> ScriptResp::toMessageImpl() && {
  std::vector<at::IValue> ivalues;
  ivalues.reserve(2);
  ivalues.emplace_back(value_);
  ivalues.emplace_back(deviceMapToC10Dict(deviceMap_));
  auto res = fromIValues(ivalues, MessageType::SCRIPT_RET);
  res->setDeviceMap(std::move(deviceMap_));
  return res;
}

std::unique_ptr<ScriptResp> ScriptResp::fromMessage(const Message& message) {
  auto values = toIValues(message, MessageType::SCRIPT_RET);
  TORCH_INTERNAL_ASSERT(
      values.size() == 2, "ScriptResp expects 2 IValues from message");
  auto deviceMap = c10DictToDeviceMap(values[1].to<c10::Dict<std::string, std::string>>());
  return std::make_unique<ScriptResp>(std::move(values[0]), std::move(deviceMap));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
