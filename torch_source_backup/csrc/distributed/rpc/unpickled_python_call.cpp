#include <torch/csrc/distributed/rpc/unpickled_python_call.h>

#include <torch/csrc/distributed/rpc/python_rpc_handler.h>

namespace torch::distributed::rpc {

UnpickledPythonCall::UnpickledPythonCall(
    const SerializedPyObj& serializedPyObj,
    bool isAsyncExecution)
    : isAsyncExecution_(isAsyncExecution) {
  auto& pythonRpcHandler = PythonRpcHandler::getInstance();
  pybind11::gil_scoped_acquire ag;
  pythonUdf_ = pythonRpcHandler.deserialize(serializedPyObj);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
UnpickledPythonCall::~UnpickledPythonCall() {
  // explicitly setting PyObject* to nullptr to prevent py::object's dtor to
  // decref on the PyObject again.
  // See Note [Destructing py::object] in python_ivalue.h
  py::gil_scoped_acquire acquire;
  pythonUdf_.dec_ref();
  pythonUdf_.ptr() = nullptr;
}

c10::intrusive_ptr<Message> UnpickledPythonCall::toMessageImpl() && {
  TORCH_INTERNAL_ASSERT(
      false, "UnpickledPythonCall does not support toMessage().");
}

const py::object& UnpickledPythonCall::pythonUdf() const {
  return pythonUdf_;
}

} // namespace torch::distributed::rpc
