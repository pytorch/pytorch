#include <torch/csrc/distributed/rpc/python_rpc_handler.h>

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

// if cachedFn is not none, assign it to local function; otherwise, import
// function from the scratch and assign it to local function;
py::object getLocalFunction(const py::object& cachedFn, const char* name) {
  py::object localFn = py::none();
  if (cachedFn.is_none()) {
    localFn =
        getFunction(py::module::import("torch.distributed.rpc.internal"), name);
  } else {
    localFn = cachedFn;
  }
  return localFn;
}

} // namespace

PythonRpcHandler::PythonRpcHandler() {
  AutoGIL ag;
  py::object module = py::module::import("torch.distributed.rpc.internal");
  pyRunFunction_ = getFunction(module, "_run_function");
  pyLoadReturnValue_ = getFunction(module, "_load_return_value");
  pySerialize_ = getFunction(module, "serialize");
}

void PythonRpcHandler::cleanup() {
  AutoGIL ag;
  pyRunFunction_ = py::none();
  pyLoadReturnValue_ = py::none();
  pySerialize_ = py::none();
}

PythonRpcHandler& PythonRpcHandler::getInstance() {
  static PythonRpcHandler handler;
  return handler;
}

std::vector<char> PythonRpcHandler::generatePythonUDFResult(
    const std::vector<char>& pickledPayload,
    const std::vector<torch::Tensor>& requestTensorTable,
    std::vector<torch::Tensor>& responseTensorTable) {
  AutoGIL ag;
  auto pargs = py::bytes(pickledPayload.data(), pickledPayload.size());
  // Get local functions first as it is possible that generatePythonUDFResult is
  // called after RpcAgent.join() where cached functions were cleaned
  // up.
  auto pyRunFunction = getLocalFunction(pyRunFunction_, "_run_function");
  auto pySerialize = getLocalFunction(pySerialize_, "serialize");
  py::tuple pres = pySerialize(pyRunFunction(pargs, requestTensorTable));
  const auto& presStr = pres[0].cast<std::string>();
  responseTensorTable = pres[1].cast<std::vector<torch::Tensor>>();
  std::vector<char> payload(presStr.begin(), presStr.end());
  return payload;
}

py::object PythonRpcHandler::loadPythonUDFResult(
    const std::vector<char>& pickledPayload,
    const std::vector<torch::Tensor>& tensorTable) {
  AutoGIL ag;
  auto pargs = py::bytes(pickledPayload.data(), pickledPayload.size());
  // Get local functions first as it is possible that loadPythonUDFResult is
  // called after RpcAgent.join() where cached functions were cleaned
  // up.
  auto pyLoadReturnValue =
      getLocalFunction(pyLoadReturnValue_, "_load_return_value");
  return pyLoadReturnValue(pargs, tensorTable);
}

py::object PythonRpcHandler::runPythonUDF(
    const SerializedPyObj& serializedObj) {
  AutoGIL ag;
  // Get local functions first as it is possible that runPythonUDF is
  // called after RpcAgent.join() where cached functions were cleaned
  // up.
  auto pyRunFunction = getLocalFunction(pyRunFunction_, "_run_function");
  return pyRunFunction(
      py::bytes(serializedObj.payload_), serializedObj.tensors_);
}

SerializedPyObj PythonRpcHandler::serialize(const py::object& obj) {
  AutoGIL ag;
  // Get local functions first as it is possible that serialize is
  // called after RpcAgent.join() where cached functions were cleaned
  // up.
  auto pySerialize = getLocalFunction(pySerialize_, "serialize");
  py::tuple t = pySerialize(obj);
  return SerializedPyObj(
      t[0].cast<std::string>(), t[1].cast<std::vector<torch::Tensor>>());
}

py::object PythonRpcHandler::deserialize(const SerializedPyObj& serializedObj) {
  AutoGIL ag;
  // Get local functions first as it is possible that deserialize is
  // called after RpcAgent.join() where cached functions were cleaned
  // up.
  auto pyLoadReturnValue =
      getLocalFunction(pyLoadReturnValue_, "_load_return_value");
  return pyLoadReturnValue(
      py::bytes(serializedObj.payload_), serializedObj.tensors_);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
