#include <torch/csrc/utils/qengines.h>

#include <c10/core/QEngine.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/QEngine.h>

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch {
namespace utils {

void addQEngine(
    at::QEngine qengine,
    const std::string& name,
    PyObject* torch_module) {
  PyObject* qengine_obj = THPQEngine_New(qengine, name);
  Py_INCREF(qengine_obj);
  if (PyModule_AddObject(torch_module, name.c_str(), qengine_obj) != 0) {
    throw python_error();
  }
}

void initializeQEngines() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) {
    throw python_error();
  }

  addQEngine(at::kNoQEngine, "no_qengine", torch_module);
  addQEngine(at::kFBGEMM, "fbgemm", torch_module);
  addQEngine(at::kQNNPACK, "qnnpack", torch_module);
}

} // namespace utils
} // namespace torch
