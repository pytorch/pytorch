#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/python_compat.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

// A macro that grabs the GIL, profiling the acquisition time. The average GIL
// acquisition time will be recorded in RpcAgent's getMetrics().
#define PROFILE_GIL_SCOPED_ACQUIRE                                       \
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime; \
  auto shouldProfileGIL =                                                \
      RpcAgent::getCurrentRpcAgent()->isGILProfilingEnabled();           \
  if (shouldProfileGIL) {                                                \
    startTime = std::chrono::high_resolution_clock::now();               \
  }                                                                      \
  pybind11::gil_scoped_acquire ag;                                       \
  if (shouldProfileGIL) {                                                \
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(    \
        std::chrono::high_resolution_clock::now() - startTime);          \
    RpcAgent::getCurrentRpcAgent()->addGilWaitTime(dur);                 \
  }

// PythonTypeResolver that inherits from Script::Resolver to
// support resolving types together with ScriptTypeParser.
struct PythonTypeResolver : public jit::Resolver {
  std::shared_ptr<jit::SugaredValue> resolveValue(
      const std::string& /* unused */,
      torch::jit::Function& /* unused */,
      const jit::SourceRange& /* unused */) override {
    TORCH_INTERNAL_ASSERT(
        false, "RPC Type resolver does not need to resolve value");
  }

  TypePtr resolveType(
      const std::string& name,
      const jit::SourceRange& /* unused */) override {
    if (name == "PyObject") {
      return PyObjectType::get();
    }
    return PythonRpcHandler::getInstance().jitCompilationUnit()->get_type(name);
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
  PROFILE_GIL_SCOPED_ACQUIRE;
  py::object module = py::module::import("torch.distributed.rpc.internal");
  pyRunFunction_ = getFunction(module, "_run_function");
  pyLoadReturnValue_ = getFunction(module, "_load_return_value");
  pySerialize_ = getFunction(module, "serialize");
  pyHandleException_ = getFunction(module, "_handle_exception");
  jitCompilationUnit_ = torch::jit::get_python_cu();
  typeParser_ = std::make_shared<jit::ScriptTypeParser>(
      std::make_shared<PythonTypeResolver>());
}

void PythonRpcHandler::cleanup() {
  PROFILE_GIL_SCOPED_ACQUIRE;
  pyRunFunction_ = py::none();
  pyLoadReturnValue_ = py::none();
  pySerialize_ = py::none();
  pyHandleException_ = py::none();
  jitCompilationUnit_ = nullptr;
  typeParser_ = nullptr;
}

PythonRpcHandler& PythonRpcHandler::getInstance() {
  // Leaky singleton to avoid module destructor race.
  static PythonRpcHandler* handler = new PythonRpcHandler();
  return *handler;
}

std::shared_ptr<torch::jit::CompilationUnit> PythonRpcHandler::
    jitCompilationUnit() {
  return jitCompilationUnit_;
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

void PythonRpcHandler::handleExceptionGILHeld(const py::object& obj) {
  TORCH_CHECK(PyGILState_Check(), "GIL should be held");
  pyHandleException_(obj);
}

TypePtr PythonRpcHandler::parseTypeFromStr(const std::string& type_str) {
  return typeParser_->parseType(type_str);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
