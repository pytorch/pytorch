#include <Python.h>
#include "test_backend.h"

#ifdef _WIN32
__declspec(dllimport)
#endif

#ifndef _WIN32
    extern "C" __attribute__((visibility("default")))
    PyObject* PyInit_libtest_jit_python();
#endif

PyObject* module;
static std::vector<PyMethodDef> methods;

PyMODINIT_FUNC PyInit_libtest_jit_python() {
  static struct PyModuleDef test_module = {
      PyModuleDef_HEAD_INIT, "libtest_jit_python", nullptr, -1, methods.data()};
  module = PyModule_Create(&test_module);

  if (!module) {
    return nullptr;
  }

  torch::jit::initTestBackendBindings(module);

  return module;
}
