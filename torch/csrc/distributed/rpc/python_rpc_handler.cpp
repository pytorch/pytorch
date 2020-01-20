#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {


// PythonTypeResolver that inherits from Script::Resolver to
// support resolving types together with ScriptTypeParser.
struct PythonTypeResolver: public jit::script::Resolver {
  std::shared_ptr<jit::script::SugaredValue> resolveValue(
      const std::string& name,
      Function& m,
      const jit::SourceRange& loc) override {
    AT_ERROR("RPC Type resolver does not need to resolve value");
  }

  TypePtr resolveType(const std::string& name, const jit::SourceRange& loc)
      override {
    if (name == "PyObject") {
      return PyObjectType::get();
    }
    auto python_cu = torch::jit::get_python_cu();
    return python_cu->get_type(name);
  }
};

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
  pybind11::gil_scoped_acquire ag;
  py::object module = py::module::import("torch.distributed.rpc.internal");
  pyRunFunction_ = getFunction(module, "_run_function");
  pyLoadReturnValue_ = getFunction(module, "_load_return_value");
  pySerialize_ = getFunction(module, "serialize");
  pyHandleException_ = getFunction(module, "_handle_exception");
  typeParser_ = std::make_shared<jit::script::ScriptTypeParser>(
                    std::make_shared<PythonTypeResolver>());
}

void PythonRpcHandler::cleanup() {
  pybind11::gil_scoped_acquire ag;
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
  pybind11::gil_scoped_acquire ag;
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
  pybind11::gil_scoped_acquire ag;
  auto pargs = py::bytes(pickledPayload.data(), pickledPayload.size());
  return pyLoadReturnValue_(pargs, tensorTable);
}

py::object PythonRpcHandler::runPythonUDF(
    const SerializedPyObj& serializedObj) {
  pybind11::gil_scoped_acquire ag;
  return pyRunFunction_(
      py::bytes(serializedObj.payload_), serializedObj.tensors_);
}

SerializedPyObj PythonRpcHandler::serialize(const py::object& obj) {
  pybind11::gil_scoped_acquire ag;
  py::tuple t = pySerialize_(obj);
  return SerializedPyObj(
      t[0].cast<std::string>(), t[1].cast<std::vector<torch::Tensor>>());
}

py::object PythonRpcHandler::deserialize(const SerializedPyObj& serializedObj) {
  pybind11::gil_scoped_acquire ag;
  return pyLoadReturnValue_(
      py::bytes(serializedObj.payload_), serializedObj.tensors_);
}

void PythonRpcHandler::handleException(const py::object& obj) {
  pybind11::gil_scoped_acquire ag;
  pyHandleException_(obj);
}

TypePtr PythonRpcHandler::parseTypeFromStr(const std::string& type_str) {
  return typeParser_->parseType(type_str);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
