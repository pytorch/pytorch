#include <torch/csrc/distributed/rpc/PythonRpcHandler.h>

namespace torch {
namespace distributed {
namespace rpc {

PyObject* PythonRpcHandler::module_ = nullptr;
PyObject* PythonRpcHandler::runUDFFunction_ = nullptr;
PyObject* PythonRpcHandler::loadResultFunction_ = nullptr;
bool PythonRpcHandler::pyInitializerCalled_ = false;

void PythonRpcHandler::init() {
  if (!Py_IsInitialized()) {
    Py_Initialize();
    pyInitializerCalled_ = true;
  }
  module_ = PyImport_ImportModule("torch.distributed.internal_rpc_utils");
  if (module_ == nullptr) {
    throw std::runtime_error("import internal_rpc_utils module is NULL");
  }
  runUDFFunction_ = PyObject_GetAttrString(module_, "run_python_udf_internal");
  if (runUDFFunction_ == nullptr) {
    throw std::runtime_error("import run_python_udf_internal is NULL");
  }
  loadResultFunction_ = PyObject_GetAttrString(
    module_, "load_python_udf_result_internal");
  if (loadResultFunction_ == nullptr) {
      throw std::runtime_error(
        "import load_python_udf_result_internal is NULL");
  }
}

void PythonRpcHandler::cleanUp() {
  Py_DECREF(module_);
  Py_DECREF(runUDFFunction_);
  Py_DECREF(loadResultFunction_);
  if(pyInitializerCalled_) {
    Py_Finalize();
  }
}

std::vector<char> PythonRpcHandler::generatePythonUDFResult(
  const Message& request) {
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();
  auto pargs =
      Py_BuildValue("(y#)", request.payload().data(), request.payload().size());
  if (pargs == nullptr) {
    throw std::runtime_error("python function args is NULL");
  }
  auto pres = PyEval_CallObject(runUDFFunction_, pargs);
  Py_DECREF(pargs);
  if (pres == nullptr) {
    throw std::runtime_error("received null result");
  }
  char* res;
  int size;
  PyArg_Parse(pres, "y#", &res, &size);
  if (res == nullptr) {
    throw std::runtime_error("serialized string is null");
  }
  std::vector<char> payload(res, res + size);
  Py_DECREF(pres);
  PyGILState_Release(gstate);
  return payload;
}

py::object PythonRpcHandler::loadPythonUDFResult(const Message& message) {
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();
  auto pargs =
      Py_BuildValue("(y#)", message.payload().data(), message.payload().size());
  auto pres = PyEval_CallObject(loadResultFunction_, pargs);
  if (pargs == nullptr) {
    throw std::runtime_error("python function args is NULL");
  }
  Py_DECREF(pargs);
  if (pres == nullptr) {
    throw std::runtime_error("recevied null result");
  }
  PyGILState_Release(gstate);
  auto a = py::cast<py::object>(pres);
  return a;
}

}
}
}
