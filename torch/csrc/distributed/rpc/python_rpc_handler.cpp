#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

py::object getFunction(const py::object& module, const char* name) {
  py::object fn = module.attr(name);
  TORCH_CHECK(
      py::isinstance<py::function>(fn),
      "attribute ",
      name,
      " is not a function");
  return fn;
}

} // namespace

PythonRpcHandler::PythonRpcHandler() {
  PROFILE_GIL_SCOPED_ACQUIRE;
  py::object module = py::module::import("torch.distributed.rpc.internal");
  pyRunFunction_ = getFunction(module, "_run_function");
  pyLoadReturnValue_ = getFunction(module, "_load_return_value");
  pySerialize_ = getFunction(module, "serialize");
  pyHandleException_ = getFunction(module, "_handle_exception");
}

void PythonRpcHandler::cleanup() {
  PROFILE_GIL_SCOPED_ACQUIRE;
  pyRunFunction_ = py::none();
  pyLoadReturnValue_ = py::none();
  pySerialize_ = py::none();
  pyHandleException_ = py::none();
}

PythonRpcHandler& PythonRpcHandler::getInstance() {
  static PythonRpcHandler handler;
  return handler;
}

std::vector<char> PythonRpcHandler::generatePythonUDFResult(
    const std::vector<char>& pickledPayload,
    const std::vector<torch::Tensor>& requestTensorTable,
    std::vector<torch::Tensor>& responseTensorTable) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  auto pargs = py::bytes(pickledPayload.data(), pickledPayload.size());
  py::tuple pres = pySerialize_(pyRunFunction_(pargs, requestTensorTable));
  const auto& presStr = pres[0].cast<std::string>();
  responseTensorTable = pres[1].cast<std::vector<torch::Tensor>>();
  std::vector<char> payload(presStr.begin(), presStr.end());
  return payload;
}

py::object PythonRpcHandler::loadPythonUDFResult(
    const std::vector<char>& pickledPayload,
    const std::vector<torch::Tensor>& tensorTable) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  auto pargs = py::bytes(pickledPayload.data(), pickledPayload.size());
  return pyLoadReturnValue_(pargs, tensorTable);
}

py::object PythonRpcHandler::runPythonUDF(
    const SerializedPyObj& serializedObj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  return pyRunFunction_(
      py::bytes(serializedObj.payload_), serializedObj.tensors_);
}

SerializedPyObj PythonRpcHandler::serialize(const py::object& obj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  py::tuple t = pySerialize_(obj);
  return SerializedPyObj(
      t[0].cast<std::string>(), t[1].cast<std::vector<torch::Tensor>>());
}

py::object PythonRpcHandler::deserialize(const SerializedPyObj& serializedObj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  return pyLoadReturnValue_(
      py::bytes(serializedObj.payload_), serializedObj.tensors_);
}

void PythonRpcHandler::handleException(const py::object& obj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  pyHandleException_(obj);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
