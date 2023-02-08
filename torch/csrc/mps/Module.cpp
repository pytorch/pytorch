#include <ATen/ATen.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSGeneratorImpl.h>

#include <torch/csrc/Generator.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

using namespace torch;

static PyObject* MPSModule_initExtension(PyObject* self, PyObject* noargs) {
#if C10_ASAN_ENABLED
  TORCH_WARN(
      "torch.mps: your pytorch binary has address sanitizer (asan) built in, "
      "asan is currently not compatible with torch.mps module, "
      "you might get unexpected behavior (eg. out of memory, crash, etc.), "
      "please rebuild pytorch without asan if you need to use this module");
#endif
  HANDLE_TH_ERRORS

  auto m = THPObjectPtr(PyImport_ImportModule("torch.mps"));
  if (!m)
    throw python_error();

  auto set_module_attr = [&](const char* name, PyObject* v) {
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto gen = at::mps::detail::getDefaultMPSGenerator();
  auto default_mps_generator =
      (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
  set_module_attr("default_mps_generator", (PyObject*)default_mps_generator);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* MPSModule_isAvailable(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::detail::getMPSHooks().hasMPS()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* MPSModule_isMacOS13orNewer(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::detail::getMPSHooks().isOnMacOS13orNewer()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* MPSModule_Synchronize(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::detail::getMPSHooks().deviceSynchronize();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)
static struct PyMethodDef _MPSModule_methods[] = {
    {"_mps_init", MPSModule_initExtension, METH_NOARGS, nullptr},
    {"_mps_synchronize", MPSModule_Synchronize, METH_NOARGS, nullptr},
    {"_is_mps_available", MPSModule_isAvailable, METH_NOARGS, nullptr},
    {"_is_mps_on_macos_13_or_newer",
     MPSModule_isMacOS13orNewer,
     METH_NOARGS,
     nullptr},
};

PyMethodDef* MPSModule_methods() {
  return _MPSModule_methods;
}