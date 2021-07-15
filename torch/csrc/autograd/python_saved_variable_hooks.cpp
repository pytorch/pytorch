#include <torch/csrc/autograd/python_saved_variable_hooks.h>

namespace py = pybind11;

namespace torch { namespace autograd {
  PySavedVariableHooks::PySavedVariableHooks(py::function &pack_hook, py::function &unpack_hook) :
    // steals the reference (we will decref ourselves)
    pack_hook_(pack_hook.release().ptr()),
    unpack_hook_(unpack_hook.release().ptr()) {}

  // NOLINTNEXTLINE(clang-diagnostic-unused-parameter)
  void PySavedVariableHooks::call_pack_hook(at::Tensor &tensor) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Hooks are not implemented yet");
  }

  at::Tensor PySavedVariableHooks::call_unpack_hook() {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Hooks are not implemented yet");
  }

  PySavedVariableHooks::~PySavedVariableHooks() {
    // If python is already dead, leak the wrapped python objects
    if (Py_IsInitialized()) {
      py::gil_scoped_acquire gil;
      Py_XDECREF(pack_hook_);
      Py_XDECREF(unpack_hook_);
    }
  }
}}
