#include <torch/csrc/distributed/rpc/python_resp.h>

#include <torch/csrc/distributed/rpc/utils.h>

#include <c10/util/C++17.h>

namespace torch {
namespace distributed {
namespace rpc {

PythonResp::PythonResp(SerializedPyObj&& serializedPyObj, DeviceMap deviceMap)
    : serializedPyObj_(std::move(serializedPyObj)), deviceMap_(std::move(deviceMap)) {}

c10::intrusive_ptr<Message> PythonResp::toMessageImpl() && {
  auto res = fromIValues(std::move(serializedPyObj_).toIValues(), MessageType::PYTHON_RET);
  res->setDeviceMap(std::move(deviceMap_));
  return res;
}

std::unique_ptr<PythonResp> PythonResp::fromMessage(const Message& message) {
  return std::make_unique<PythonResp>(std::move(SerializedPyObj::fromIValues(toIValues(message, MessageType::PYTHON_RET))), DeviceMap());
}

const SerializedPyObj& PythonResp::serializedPyObj() const {
  return serializedPyObj_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
