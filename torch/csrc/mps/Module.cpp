#include <ATen/ATen.h>
#include <c10/util/CallOnce.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_numbers.h>

// pthread.h is included for tracking bad forks
#ifndef WIN32
#include <pthread.h>
#endif

namespace torch {
namespace mps {

namespace {
// True for children forked after mps init
static bool in_bad_fork = false;

// Called in the forked child if mps has already been initialized
static void forked_mps_child() {
  in_bad_fork = true;
}

// Should be called before the first mps call.
static void track_bad_mps_fork() {
#ifndef WIN32
  static c10::once_flag flag;
  c10::call_once(
      flag, [] { pthread_atfork(nullptr, nullptr, forked_mps_child); });
#endif
}
} // namespace

static PyObject* MPSModule_isInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_getDefaultMPSGenerator(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  track_bad_mps_fork();
  return THPGenerator_initDefaultGenerator(
      at::detail::getMPSHooks().getDefaultMPSGenerator());
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_isAvailable(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  track_bad_mps_fork();
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
    {"_mps_is_in_bad_fork", MPSModule_isInBadFork, METH_NOARGS, nullptr},
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
