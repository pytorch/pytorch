#include <ATen/ATen.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_numbers.h>

namespace torch {
namespace mps {

static PyObject* MPSModule_getDefaultMPSGenerator(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPGenerator_initDefaultGenerator(
      at::detail::getMPSHooks().getDefaultMPSGenerator());
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_isAvailable(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::detail::getMPSHooks().hasMPS()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_isMacOS13orNewer(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::detail::getMPSHooks().isOnMacOS13orNewer()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_synchronize(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::detail::getMPSHooks().deviceSynchronize();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)
static struct PyMethodDef _MPSModule_methods[] = {
    {"_mps_synchronize", MPSModule_synchronize, METH_NOARGS, nullptr},
    {"_is_mps_available", MPSModule_isAvailable, METH_NOARGS, nullptr},
    {"_is_mps_on_macos_13_or_newer",
     MPSModule_isMacOS13orNewer,
     METH_NOARGS,
     nullptr},
    {"_mps_get_default_generator",
     MPSModule_getDefaultMPSGenerator,
     METH_NOARGS,
     nullptr},
    {nullptr}};

PyMethodDef* python_functions() {
  return _MPSModule_methods;
}

} // namespace mps
} // namespace torch
