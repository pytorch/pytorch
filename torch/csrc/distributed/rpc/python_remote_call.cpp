#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonRemoteCall::PythonRemoteCall(
    SerializedPyObj&& serializedPyObj,
    at::IValue retRRefId,
    at::IValue retForkId)
    : serializedPyObj_(std::move(serializedPyObj)),
      retRRefId_(std::move(retRRefId)),
      retForkId_(std::move(retForkId)) {}

Message PythonRemoteCall::toMessage() && {
  isInRpcCall = true;
  std::vector<IValue> ivalues = std::move(serializedPyObj_).toIValues();
  ivalues.emplace_back(retRRefId_);
  ivalues.emplace_back(retForkId_);

  std::vector<torch::Tensor> tensor_table;
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);

  isInRpcCall = false;
  return Message(
      std::move(payload),
      std::move(tensor_table),
      MessageType::PYTHON_REMOTE_CALL);
}

std::unique_ptr<PythonRemoteCall> PythonRemoteCall::fromMessage(
    const Message& message) {
  auto payload = static_cast<const char*>(message.payload().data());
  auto payload_size = message.payload().size();

  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      &message.tensors());
  auto values = value.toTuple()->elements();

  // remove the last element from values and convert it back to an RRef
  auto retForkId = std::move(values.back());
  values.pop_back();
  auto retRRefId = std::move(values.back());
  values.pop_back();
  auto serializedPyObj = SerializedPyObj::fromIValues(std::move(values));

  return std::make_unique<PythonRemoteCall>(
      std::move(serializedPyObj), std::move(retRRefId), std::move(retForkId));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
