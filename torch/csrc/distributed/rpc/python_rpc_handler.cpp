#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/python_compat.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

constexpr auto kInternalModule = "torch.distributed.rpc.internal";

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
  } // NOLINT

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

void cleanupPyObj(py::object& obj) {
  obj.dec_ref();
  // explicitly setting PyObject* to nullptr to prevent py::object's dtor to
  // decref on the PyObject again.
  // See Note [Destructing py::object] in python_ivalue.h
  obj.ptr() = nullptr;
}

} // namespace

void PythonRpcHandler::init() {
  std::lock_guard<std::mutex> guard(init_lock_);
  if (!initialized_) {
    PROFILE_GIL_SCOPED_ACQUIRE;
    py::object rpcInternal = py::module::import(kInternalModule);
    py::object rpcApi = py::module::import("torch.distributed.rpc.api");
    py::object rrefProxy =
        py::module::import("torch.distributed.rpc.rref_proxy");

    pyRunFunction_ = getFunction(rpcInternal, "_run_function");
    pySerialize_ = getFunction(rpcInternal, "serialize");
    pyDeserialize_ = getFunction(rpcInternal, "deserialize");
    pyHandleException_ = getFunction(rpcInternal, "_handle_exception");

    rrefTypeFunctions_.onOwner_ = getFunction(rpcApi, "_rref_typeof_on_owner");
    rrefTypeFunctions_.onUser_ = getFunction(rpcApi, "_rref_typeof_on_user");

    rrefProxyFunctions_.rpcSync_ = getFunction(rpcApi, "rpc_sync");
    rrefProxyFunctions_.rpcAsync_ = getFunction(rpcApi, "rpc_async");
    rrefProxyFunctions_.remote_ = getFunction(rpcApi, "remote");
    rrefProxyFunctions_.rrefProxyCtor_ = getFunction(rrefProxy, "RRefProxy");

    jitCompilationUnit_ = torch::jit::get_python_cu();
    typeParser_ = std::make_shared<jit::ScriptTypeParser>(
        std::make_shared<PythonTypeResolver>());
    initialized_ = true;
  }
}

PythonRpcHandler::PythonRpcHandler() : initialized_(false) {}

void PythonRpcHandler::cleanup() {
  std::lock_guard<std::mutex> guard(init_lock_);
  PROFILE_GIL_SCOPED_ACQUIRE;
  cleanupPyObj(pyRunFunction_);
  cleanupPyObj(pySerialize_);
  cleanupPyObj(pyDeserialize_);
  cleanupPyObj(pyHandleException_);

  cleanupPyObj(rrefProxyFunctions_.rpcSync_);
  cleanupPyObj(rrefProxyFunctions_.rpcAsync_);
  cleanupPyObj(rrefProxyFunctions_.remote_);
  cleanupPyObj(rrefProxyFunctions_.rrefProxyCtor_);

  jitCompilationUnit_ = nullptr;
  typeParser_ = nullptr;
  initialized_ = false;
}

PythonRpcHandler& PythonRpcHandler::getInstance() {
  // A thread could hold GIL when calling PythonRpcHandler::getInstance(),
  // meantime another thread could have been doing static data
  // initialization by calling `new PythonRpcHandler()`, inside of which GIL is
  // also required. Static data initialization is thread-safe, so the thread
  // holding the GIL will wait for the other thread to finish static data
  // initializating before going forward. Because the initialization can't
  // proceed without GIL, there is a deadlock. We ask the calling thread to
  // release GIL to avoid this situation.
  TORCH_INTERNAL_ASSERT(!PyGILState_Check());
  // Leaky singleton to avoid module destructor race.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static PythonRpcHandler* handler = new PythonRpcHandler();
  handler->init();
  return *handler;
}

std::shared_ptr<torch::jit::CompilationUnit> PythonRpcHandler::
    jitCompilationUnit() {
  return jitCompilationUnit_;
}

py::object PythonRpcHandler::runPythonUdf(const py::object& pythonUdf) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  // Throw a descriptive error message if pyRunFunction_ is already cleaned up.
  TORCH_INTERNAL_ASSERT(
      !pyRunFunction_.is_none(),
      "Cannot run python UDF since pyRunFunction_ is None. Check if python RPC "
      "handler is already cleaned up.");
  return pyRunFunction_(pythonUdf);
}

SerializedPyObj PythonRpcHandler::serialize(const py::object& obj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  py::tuple t = pySerialize_(obj);
  return SerializedPyObj(
      t[0].cast<std::string>(), t[1].cast<std::vector<torch::Tensor>>());
}

py::object PythonRpcHandler::deserialize(const SerializedPyObj& serializedObj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  // NB: pyDeserialize_ can return an AttributeError if the deserialize() Python
  // function fails. Functions consuming the result needs to handle such error
  // properly.
  return pyDeserialize_(
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

bool PythonRpcHandler::isRemoteException(const py::object& obj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  auto type = obj.get_type();
  auto moduleName = type.attr("__module__").cast<std::string>();
  auto qualName = type.attr("__qualname__").cast<std::string>();
  return moduleName.compare(kInternalModule) == 0 &&
      qualName.compare("RemoteException") == 0;
}

TypePtr PythonRpcHandler::parseTypeFromStr(const std::string& type_str) {
  return typeParser_->parseType(type_str);
}

const PythonRpcHandler::RRefProxyFunctions& PythonRpcHandler::
    getRRefProxyFunctions() const {
  return rrefProxyFunctions_;
}

const PythonRpcHandler::RRefTypeFunctions& PythonRpcHandler::
    getRRefTypeFunctions() const {
  return rrefTypeFunctions_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
