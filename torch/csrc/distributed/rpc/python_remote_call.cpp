#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonRemoteCall::PythonRemoteCall(
    SerializedPyObj&& serializedPyObj,
    at::IValue retRRefId,
    at::IValue retForkId,
    DeviceMap&& deviceMap,
    const bool isAsyncExecution)
    : serializedPyObj_(std::move(serializedPyObj)),
      retRRefId_(std::move(retRRefId)),
      retForkId_(std::move(retForkId)),
      deviceMap_(deviceMap),
      isAsyncExecution_(isAsyncExecution) {}

c10::intrusive_ptr<Message> PythonRemoteCall::toMessageImpl() && {
  std::vector<IValue> ivalues = std::move(serializedPyObj_).toIValues();
  ivalues.emplace_back(retRRefId_);
  ivalues.emplace_back(retForkId_);
  ivalues.emplace_back(isAsyncExecution_);
  ivalues.emplace_back(deviceMapToC10Dict(deviceMap_));

  std::vector<torch::Tensor> tensor_table;
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);

  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(tensor_table),
      MessageType::PYTHON_REMOTE_CALL,
      std::move(deviceMap_));
}

std::unique_ptr<PythonRemoteCall> PythonRemoteCall::fromMessage(
    const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  auto values = value.toTuple()->elements();

  // remove the last elements from values and convert it back to an RRef
  TORCH_INTERNAL_ASSERT(
      values.size() >= 3,
      "Expect at least 3 elements in the unpickled values, but got ",
      values.size());

  auto c10DeviceMap = values.back().to<c10::Dict<std::string, std::string>>();
  std::unordered_map<c10::Device, c10::Device> deviceMap =
      c10DictToDeviceMap(c10DeviceMap);
  values.pop_back();

  bool isAsyncExecution = values.back().toBool();
  values.pop_back();
  auto retForkId = std::move(values.back());
  values.pop_back();
  auto retRRefId = std::move(values.back());
  values.pop_back();
  auto serializedPyObj = SerializedPyObj::fromIValues(std::move(values));

  return std::make_unique<PythonRemoteCall>(
      std::move(serializedPyObj),
      std::move(retRRefId),
      std::move(retForkId),
      std::move(deviceMap),
      isAsyncExecution);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
