#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch::distributed::rpc {

PythonRemoteCall::PythonRemoteCall(
    SerializedPyObj&& serializedPyObj,
    at::IValue retRRefId,
    at::IValue retForkId,
    const bool isAsyncExecution)
    : serializedPyObj_(std::move(serializedPyObj)),
      retRRefId_(std::move(retRRefId)),
      retForkId_(std::move(retForkId)),
      isAsyncExecution_(isAsyncExecution) {}

c10::intrusive_ptr<Message> PythonRemoteCall::toMessageImpl() && {
  std::vector<IValue> ivalues = std::move(serializedPyObj_).toIValues();
  ivalues.emplace_back(retRRefId_);
  ivalues.emplace_back(retForkId_);
  ivalues.emplace_back(isAsyncExecution_);

  std::vector<torch::Tensor> tensor_table;
  auto payload =
      jit::pickle(c10::ivalue::Tuple::create(ivalues), &tensor_table);

  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(tensor_table),
      MessageType::PYTHON_REMOTE_CALL);
}

std::unique_ptr<PythonRemoteCall> PythonRemoteCall::fromMessage(
    const Message& message) {
  auto payload = message.payload().data();
  auto payload_size = message.payload().size();

  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  auto values = value.toTupleRef().elements().vec();

  // remove the last elements from values and convert it back to an RRef
  TORCH_INTERNAL_ASSERT(
      values.size() > 3,
      "Expect at least 4 elements in the unpickled values, but got ",
      values.size());
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
      isAsyncExecution);
}

} // namespace torch::distributed::rpc
