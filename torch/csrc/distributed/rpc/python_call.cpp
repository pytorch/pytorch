#include <torch/csrc/distributed/rpc/python_call.h>

#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <c10/util/C++17.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonCall::PythonCall(
    SerializedPyObj&& serializedPyObj,
    DeviceMap&& deviceMap,
    bool isAsyncExecution)
    : serializedPyObj_(std::move(serializedPyObj)),
      deviceMap_(std::move(deviceMap)),
      isAsyncExecution_(isAsyncExecution) {}

c10::intrusive_ptr<Message> PythonCall::toMessageImpl() && {
  std::vector<IValue> ivalues = std::move(serializedPyObj_).toIValues();
  ivalues.emplace_back(isAsyncExecution_);

  // Convert deviceMap to c10::Dict for serialization.
  ivalues.emplace_back(deviceMapToC10Dict(deviceMap_));

  // TODO(pbelevich): replace with fromIValues
  std::vector<torch::Tensor> tensor_table;
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);

  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(tensor_table),
      MessageType::PYTHON_CALL,
      std::move(deviceMap_));
}

std::unique_ptr<PythonCall> PythonCall::fromMessage(const Message& message) {
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
      values.size() >= 1,
      "Failed to convert an RPC message to PythonCall, the payload should at "
      "least contain one byte indicating whether this is an async function, "
      "but got payload of size ",
      values.size());

  auto c10DeviceMap = values.back().to<c10::Dict<std::string, std::string>>();
  // Convert to regular map.
  std::unordered_map<c10::Device, c10::Device> deviceMap =
      c10DictToDeviceMap(c10DeviceMap);
  values.pop_back();

  bool isAsyncExecution = values.back().toBool();
  values.pop_back();
  auto serializedPyObj = SerializedPyObj::fromIValues(std::move(values));

  return std::make_unique<PythonCall>(
      std::move(serializedPyObj), std::move(deviceMap), isAsyncExecution);
}

const SerializedPyObj& PythonCall::serializedPyObj() const {
  return serializedPyObj_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
