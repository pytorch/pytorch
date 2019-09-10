#include <torch/csrc/utils/qbackends.h>

#include <c10/core/QBackend.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/QBackend.h>

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch {
namespace utils {

#define _ADD_QBACKEND(qbackend, name)                                \
  {                                                                  \
    PyObject* qbackend_obj = THPQBackend_New(qbackend, name);        \
    Py_INCREF(qbackend_obj);                                         \
    if (PyModule_AddObject(torch_module, name, qbackend_obj) != 0) { \
      throw python_error();                                          \
    }                                                                \
  }

void initializeQBackends() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) {
    throw python_error();
  }

  _ADD_QBACKEND(at::kFBGEMM, "fbgemm");
  _ADD_QBACKEND(at::kQNNPACK, "qnnpack");
}

} // namespace utils
} // namespace torch
