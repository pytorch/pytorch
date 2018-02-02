#pragma once

#include <Python.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/autograd/python_variable.h"

namespace py = pybind11;

namespace pybind11 { namespace detail {

// handle Tensor <-> at::Tensor conversions
// Python Variables are unpacked into Tensors
template <> struct type_caster<at::Tensor> {
public:
  PYBIND11_TYPE_CASTER(at::Tensor, _("at::Tensor"));

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (THPVariable_Check(obj)) {
      value = ((THPVariable*)obj)->cdata.data();
      return true;
    }
    try {
      value = torch::createTensor(obj);
    } catch (std::exception& e) {
      return false;
    }
    return true;
  }
  static handle cast(at::Tensor src, return_value_policy /* policy */, handle /* parent */) {
    return handle(torch::createPyObject(src));
  }
};

}} // namespace pybind11::detail
