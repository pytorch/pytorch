#include <ATen/SavedTensorHooks.h>
#include <torch/csrc/autograd/python_saved_variable_hooks.h>

#include <torch/csrc/THP.h>

namespace py = pybind11;

namespace torch {
namespace autograd {
PySavedVariableHooks::PySavedVariableHooks(
    py::function& pack_hook,
    py::function& unpack_hook)
    : // steals the reference (we will decref ourselves)
      pack_hook_(pack_hook.release().ptr()),
      unpack_hook_(unpack_hook.release().ptr()) {}

// We don't use pybind for call_pack_hook and call_unpack_hook to avoid
// https://github.com/pytorch/pytorch/issues/34172
void PySavedVariableHooks::call_pack_hook(const at::Tensor& tensor) {
  py::gil_scoped_acquire acquire;
  THPObjectPtr obj(THPVariable_Wrap(tensor));
  THPObjectPtr packed(
      PyObject_CallFunctionObjArgs(pack_hook_, obj.get(), nullptr));
  if (!packed) {
    throw python_error();
  }
  data_ = packed.release();
  // obj is decrefed on exit, packed has their references stolen
  // pack_hook_ and data_ will be manually decrefed when the saved variable is
  // released
}

at::Tensor PySavedVariableHooks::call_unpack_hook() {
  py::gil_scoped_acquire acquire;
  THPObjectPtr res(PyObject_CallFunctionObjArgs(unpack_hook_, data_, nullptr));
  if (!res) {
    throw python_error();
  }
  TORCH_CHECK_TYPE(
      THPVariable_Check(res),
      "Output of saved tensor unpack_hook expected to be a Tensor but got result of type ",
      THPUtils_typename(res));
  return THPVariable_Unpack(res);
  // res is decrefed on exit
  // unpack_hook_ will be manually decrefed when the saved variable is released
}

// NOLINTNEXTLINE(bugprone-exception-escape)
PySavedVariableHooks::~PySavedVariableHooks() {
  // If python is already dead, leak the wrapped python objects
  if (Py_IsInitialized()) {
    py::gil_scoped_acquire gil;
    Py_XDECREF(pack_hook_);
    Py_XDECREF(unpack_hook_);
    Py_XDECREF(data_);
  }
}

void PyDefaultSavedVariableHooks::push_hooks(
    py::function& pack_hook,
    py::function& unpack_hook) {
  at::SavedTensorDefaultHooks::lazy_initialize();
  at::SavedTensorDefaultHooks::push_hooks(
      pack_hook.release().ptr(), unpack_hook.release().ptr());
}

void PyDefaultSavedVariableHooks::pop_hooks() {
  auto [pack_hook, unpack_hook] = at::SavedTensorDefaultHooks::get_hooks();
  TORCH_INTERNAL_ASSERT(pack_hook != nullptr && unpack_hook != nullptr);
  if (Py_IsInitialized()) {
    py::gil_scoped_acquire gil;
    Py_XDECREF(pack_hook);
    Py_XDECREF(unpack_hook);
  }
  at::SavedTensorDefaultHooks::pop_hooks();
}

std::unique_ptr<SavedVariableHooks> PyDefaultSavedVariableHooks::get_hooks() {
  auto [pack_hook, unpack_hook] = at::SavedTensorDefaultHooks::get_hooks();
  if (!pack_hook || !unpack_hook) {
    return nullptr;
  }
  py::gil_scoped_acquire gil;
  py::function pack_hook_ = py::reinterpret_borrow<py::function>(pack_hook);
  py::function unpack_hook_ = py::reinterpret_borrow<py::function>(unpack_hook);
  return std::make_unique<PySavedVariableHooks>(pack_hook_, unpack_hook_);
}

} // namespace autograd
} // namespace torch
