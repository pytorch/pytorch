#include <ATen/ATen.h>
#include <c10/util/CallOnce.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

// pthread.h is included for tracking bad forks
#ifndef WIN32
#include <pthread.h>
#endif

namespace torch::mps {

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

static PyObject* MPSModule_isMacOSorNewer(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  size_t major = 0;
  size_t minor = 0;
  if (!PyArg_ParseTuple(args, "LL", &major, &minor)) {
    return nullptr;
  }
  if (at::detail::getMPSHooks().isOnMacOSorNewer(major, minor)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_deviceSynchronize(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::detail::getMPSHooks().deviceSynchronize();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::detail::getMPSHooks().emptyCache();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_setMemoryFraction(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkDouble(args), "invalid argument to setMemoryFraction()");
  double fraction = THPUtils_unpackDouble(args);
  at::detail::getMPSHooks().setMemoryFraction(fraction);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_currentAllocatedMemory(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(
      at::detail::getMPSHooks().getCurrentAllocatedMemory());
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_driverAllocatedMemory(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(
      at::detail::getMPSHooks().getDriverAllocatedMemory());
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_recommendedMaxMemory(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(
      at::detail::getMPSHooks().getRecommendedMaxMemory());
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_profilerStartTrace(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* mode_string_o = nullptr;
  PyObject* wait_until_completed_string_o = nullptr;
  if (!PyArg_ParseTuple(
          args, "OO", &mode_string_o, &wait_until_completed_string_o)) {
    return nullptr;
  }
  const std::string mode = THPUtils_unpackString(mode_string_o);
  const bool waitUntilCompleted =
      THPUtils_unpackBool(wait_until_completed_string_o);
  at::detail::getMPSHooks().profilerStartTrace(mode, waitUntilCompleted);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_profilerStopTrace(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::detail::getMPSHooks().profilerStopTrace();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_acquireEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const bool enable_timing = THPUtils_unpackBool(args);
  return THPUtils_packUInt32(
      at::detail::getMPSHooks().acquireEvent(enable_timing));
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_releaseEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  at::detail::getMPSHooks().releaseEvent(event_id);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_recordEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  at::detail::getMPSHooks().recordEvent(event_id);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_waitForEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  at::detail::getMPSHooks().waitForEvent(event_id);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_synchronizeEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  at::detail::getMPSHooks().synchronizeEvent(event_id);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_queryEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const uint32_t event_id = THPUtils_unpackUInt32(args);

  if (at::detail::getMPSHooks().queryEvent(event_id)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_elapsedTimeOfEvents(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* start_event_o = nullptr;
  PyObject* end_event_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &start_event_o, &end_event_o)) {
    return nullptr;
  }
  const uint32_t start_event_id = THPUtils_unpackUInt32(start_event_o);
  const uint32_t end_event_id = THPUtils_unpackUInt32(end_event_o);
  return PyFloat_FromDouble(at::detail::getMPSHooks().elapsedTimeOfEvents(
      start_event_id, end_event_id));
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays, *-global-variables)
static struct PyMethodDef _MPSModule_methods[] = {
    {"_mps_deviceSynchronize",
     MPSModule_deviceSynchronize,
     METH_NOARGS,
     nullptr},
    {"_mps_is_in_bad_fork", MPSModule_isInBadFork, METH_NOARGS, nullptr},
    {"_mps_is_available", MPSModule_isAvailable, METH_NOARGS, nullptr},
    {"_mps_is_on_macos_or_newer",
     MPSModule_isMacOSorNewer,
     METH_VARARGS,
     nullptr},
    {"_mps_get_default_generator",
     MPSModule_getDefaultMPSGenerator,
     METH_NOARGS,
     nullptr},
    {"_mps_emptyCache", MPSModule_emptyCache, METH_NOARGS, nullptr},
    {"_mps_setMemoryFraction", MPSModule_setMemoryFraction, METH_O, nullptr},
    {"_mps_currentAllocatedMemory",
     MPSModule_currentAllocatedMemory,
     METH_NOARGS,
     nullptr},
    {"_mps_driverAllocatedMemory",
     MPSModule_driverAllocatedMemory,
     METH_NOARGS,
     nullptr},
    {"_mps_recommendedMaxMemory",
     MPSModule_recommendedMaxMemory,
     METH_NOARGS,
     nullptr},
    {"_mps_profilerStartTrace",
     MPSModule_profilerStartTrace,
     METH_VARARGS,
     nullptr},
    {"_mps_profilerStopTrace",
     MPSModule_profilerStopTrace,
     METH_NOARGS,
     nullptr},
    {"_mps_acquireEvent", MPSModule_acquireEvent, METH_O, nullptr},
    {"_mps_releaseEvent", MPSModule_releaseEvent, METH_O, nullptr},
    {"_mps_recordEvent", MPSModule_recordEvent, METH_O, nullptr},
    {"_mps_waitForEvent", MPSModule_waitForEvent, METH_O, nullptr},
    {"_mps_synchronizeEvent", MPSModule_synchronizeEvent, METH_O, nullptr},
    {"_mps_queryEvent", MPSModule_queryEvent, METH_O, nullptr},
    {"_mps_elapsedTimeOfEvents",
     MPSModule_elapsedTimeOfEvents,
     METH_VARARGS,
     nullptr},
    {nullptr}};

PyMethodDef* python_functions() {
  return _MPSModule_methods;
}

} // namespace torch::mps
