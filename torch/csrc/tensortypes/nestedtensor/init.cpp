#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/ATen.h>

#include <torch/csrc/tensortypes/nestedtensor/nary.h>

#include <vector>
#include <iostream>

namespace torch {
namespace tensortypes {
namespace nestedtensor {

PyObject* nestedtensor_init(PyObject* _unused) {
  C10_LOG_API_USAGE_ONCE("tensor_list.python.import");
  auto nestedtensor_module = THPObjectPtr(PyImport_ImportModule("torch.tensortypes.nestedtensor"));
  if (!nestedtensor_module) {
    throw python_error();
  }

  auto m = py::handle(nestedtensor_module).cast<py::module>();

  Py_RETURN_TRUE;
}
// tensortypes methods on torch._C
static PyMethodDef methods[] = {
    {"_nestedtensor_init", (PyCFunction)nestedtensor_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace nestedtensor
} // namespace tensortypes
} // namespace torch
