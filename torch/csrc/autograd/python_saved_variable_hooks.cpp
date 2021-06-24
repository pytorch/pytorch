#include <torch/csrc/autograd/python_saved_variable_hooks.h>
#include <torch/csrc/autograd/python_variable.h>

namespace torch { namespace autograd {
  // PySavedVariableHooks::PySavedVariableHooks(py::function &pack_hook, py::function &unpack_hook) : pack_hook_(pack_hook), unpack_hook_(unpack_hook){};
  PySavedVariableHooks::PySavedVariableHooks(PyObject *pack_hook, PyObject *unpack_hook) : pack_hook_(pack_hook), unpack_hook_(unpack_hook){};
  // PySavedVariableHooks::PySavedVariableHooks(py::function &pack_hook, py::function &unpack_hook) {
  //   pack_hook_ = pack_hook;
  //   unpack_hook_ = unpack_hook;
  // }

  PyObject* PySavedVariableHooks::call_pack_hook(at::Tensor tensor) {
    return pack_hook_(tensor).release().ptr();
  };

  at::Tensor PySavedVariableHooks::call_unpack_hook(PyObject* obj) {
    return THPVariable_Unpack(unpack_hook_(obj).release().ptr());
  };


}}
