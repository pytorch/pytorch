#include <torch/csrc/distributed/rpc/unpickled_python_call.h>

#include <c10/util/C++17.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>

namespace torch {
namespace distributed {
namespace rpc {

UnpickledPythonCall::UnpickledPythonCall(
    const SerializedPyObj& serializedPyObj,
    bool isAsyncExecution)
    : isAsyncExecution_(isAsyncExecution) {
  auto& pythonRpcHandler = PythonRpcHandler::getInstance();
  pybind11::gil_scoped_acquire ag;
  pythonUdf_ = pythonRpcHandler.deserialize(serializedPyObj);
}

UnpickledPythonCall::~UnpickledPythonCall() {
  // explicitly setting PyObject* to nullptr to prevent py::object's dtor to
  // decref on the PyObject again.
  // See Note [Destructing py::object] in python_ivalue.h
  std::cout << "==== destructing UnpickledPythonCall\n" << std::flush;
  py::gil_scoped_acquire acquire;
  std::cout << "==== destructing UnpickledPythonCall acquired GIL\n" << std::flush;

  pythonUdf_.dec_ref();
  std::cout << "==== destructing UnpickledPythonCall derefed\n" << std::flush;

  pythonUdf_.ptr() = nullptr;
  std::cout << "==== done destructing UnpickledPythonCall\n" << std::flush;
}

Message UnpickledPythonCall::toMessageImpl() && {
  TORCH_INTERNAL_ASSERT(
      false, "UnpickledPythonCall does not support toMessage().");
}

const py::object& UnpickledPythonCall::pythonUdf() const {
  return pythonUdf_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
