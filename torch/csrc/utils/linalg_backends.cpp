#include <torch/csrc/utils/linalg_backends.h>

#include <ATen/LinalgBackend.h>

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/LinalgBackend.h>

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>


namespace torch {
namespace utils {

#define _ADD_LINALG_BACKEND(format)                                             \
  {                                                                             \
    std::string name = at::LinalgBackendToString(format);                       \
    PyObject* linalg_backend = THPLinalgBackend_New(format);                    \
    Py_INCREF(linalg_backend);                                                  \
    if (PyModule_AddObject(torch_module, name.c_str(), linalg_backend) != 0) {  \
      throw python_error();                                                     \
    }                                                                           \
  }

void initializeLinalgBackends() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) {
    throw python_error();
  }

  _ADD_LINALG_BACKEND(at::LinalgBackend::Default);
  _ADD_LINALG_BACKEND(at::LinalgBackend::Cusolver);
  _ADD_LINALG_BACKEND(at::LinalgBackend::Magma);
}

} // namespace utils
} // namespace torch
