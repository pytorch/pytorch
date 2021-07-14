#include <torch/csrc/autograd/python_saved_variable_hooks.h>

#include <torch/csrc/THP.h>

namespace py = pybind11;

namespace torch { namespace autograd {
  PySavedVariableHooks::PySavedVariableHooks(py::function &pack_hook, py::function &unpack_hook) :
    // steals the reference (we will decref ourselves)
    pack_hook_(pack_hook.release().ptr()),
    unpack_hook_(unpack_hook.release().ptr()) {}

  void PySavedVariableHooks::call_pack_hook(at::Tensor &tensor) {
    py::gil_scoped_acquire acquire;
    auto pack_hook = py::reinterpret_borrow<py::function>(pack_hook_); // borrows the reference to the hook
    auto wrapped = THPVariable_Wrap(tensor); // here, we own the reference
    py::object obj = py::reinterpret_steal<py::object>(wrapped); // obj steals the reference
    py::object packed = pack_hook(obj); // packed is a new object with a reference
    data_ = packed.release().ptr(); // steals the reference
    // pack_hook, obj are decrefed on exit, but we will manually decref data_ when the saved variable is released
    // pack_hook_ is left as is
  }

  at::Tensor PySavedVariableHooks::call_unpack_hook() {
    py::gil_scoped_acquire acquire;
    auto unpack_hook = py::reinterpret_borrow<py::function>(unpack_hook_);
    py::object obj = py::cast<py::object>(data_); // copies the content of data_
    py::object res = unpack_hook(obj); // new object, comes with a reference
    PyObject* ptr = res.ptr(); // borrows from, we only need this to be alive as long as res is alive
    TORCH_CHECK_TYPE(THPVariable_Check(ptr), "Output of saved tensor unpack_hook expected to be a Tensor but got result of type ", THPUtils_typename(ptr));
    return THPVariable_Unpack(ptr);
    // unpack_hook, obj and res are decrefed on exit
    // unpack_hook_ is left as is
  }

  PySavedVariableHooks::~PySavedVariableHooks() {
    // If python is already dead, leak the wrapped python objects
    if (Py_IsInitialized()) {
      py::gil_scoped_acquire gil;
      Py_XDECREF(pack_hook_);
      Py_XDECREF(unpack_hook_);
      Py_XDECREF(data_);
    }
  }
}}
