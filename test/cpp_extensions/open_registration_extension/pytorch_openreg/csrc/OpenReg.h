#pragma once

#include <torch/csrc/utils/pybind.h>

namespace openreg {

using openreg_ptr_t = uint64_t;

void set_impl_factory(PyObject* factory);
py::function get_method(const char* name);

static constexpr char kFreeMethod[] = "free";
static constexpr char kHostFreeMethod[] = "hostFree";

template <const char* name>
static void ReportAndDelete(void* ptr) {
  if (!ptr || !Py_IsInitialized()) {
    return;
  }

  py::gil_scoped_acquire acquire;

  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  // Always stash, this will be a no-op if there is no error
  PyErr_Fetch(&type, &value, &traceback);

  TORCH_CHECK(
      get_method(name)(reinterpret_cast<openreg_ptr_t>(ptr)).cast<bool>(),
      "Failed to free memory pointer at ",
      ptr);

  // If that user code raised an error, just print it without raising it
  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  // Restore the original error
  PyErr_Restore(type, value, traceback);
}

} // namespace openreg
