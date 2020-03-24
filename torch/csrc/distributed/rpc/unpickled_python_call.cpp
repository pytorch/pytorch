#include <torch/csrc/distributed/rpc/unpickled_python_call.h>

#include <c10/util/C++17.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>

namespace torch {
namespace distributed {
namespace rpc {

UnpickledPythonCall::UnpickledPythonCall(
    const SerializedPyObj& serializedPyObj) {
  auto& pythonRpcHandler = PythonRpcHandler::getInstance();
  pybind11::gil_scoped_acquire ag;
  pythonUdf_ = pythonRpcHandler.deserialize(serializedPyObj);
}

Message UnpickledPythonCall::toMessageImpl() && {
  TORCH_INTERNAL_ASSERT(
      false, "UnpickledPythonCall does not support toMessage().");
}

py::object UnpickledPythonCall::movePythonUdf() && {
  return std::move(pythonUdf_);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
