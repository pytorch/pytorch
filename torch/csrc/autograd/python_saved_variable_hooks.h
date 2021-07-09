#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/saved_variable_hooks.h>
#include <torch/csrc/python_headers.h>
#include <ATen/ATen.h>

namespace py = pybind11;

namespace torch { namespace autograd {

struct TORCH_API PySavedVariableHooks : public SavedVariableHooks {
    PySavedVariableHooks(py::function &pack_hook, py::function &unpack_hook) : pack_hook_(pack_hook), unpack_hook_(unpack_hook){}

    void call_pack_hook(at::Tensor &tensor) override {
      py::gil_scoped_acquire acquire;
      auto wrapped = THPVariable_Wrap(tensor); // here, we own the reference
      py::object obj = py::reinterpret_steal<py::object>(wrapped); // obj steals the reference
      py::object packed = pack_hook_(obj); // packed is a new object with a reference
      data_ = packed.release().ptr(); // steals the reference
      // obj is decrefed on exit, but we will manually decref data_ when the saved variable is released
    }

    at::Tensor call_unpack_hook() override {
      py::gil_scoped_acquire acquire;
      py::object obj = py::cast<py::object>(data_); // copies the content of data_
      py::object res = unpack_hook_(obj); // new object, comes with a reference
      return THPVariable_Unpack(res.ptr()); // borrows from res, we only need the argument as long as res is alive
      // obj and res are decrefed on exit
    }

    ~PySavedVariableHooks() override {
      if (data_) {
        Py_DECREF(data_);
      }
    };

  private:
    py::function pack_hook_;
    py::function unpack_hook_;
    PyObject* data_ = nullptr;
};

}}
