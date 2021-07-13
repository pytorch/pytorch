#include <torch/csrc/autograd/python_saved_variable_hooks.h>

namespace py = pybind11;

namespace torch { namespace autograd {
  PySavedVariableHooks::PySavedVariableHooks(py::function &pack_hook, py::function &unpack_hook) : pack_hook_(pack_hook), unpack_hook_(unpack_hook){}

  void PySavedVariableHooks::call_pack_hook(at::Tensor &tensor) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Hooks are not implemented yet");
  }

  at::Tensor PySavedVariableHooks::call_unpack_hook() {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Hooks are not implemented yet");
  }

  PySavedVariableHooks::~PySavedVariableHooks() = default;
}}
