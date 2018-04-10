#pragma once

#include <Python.h>

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/autograd/python_variable.h"

#include <stdexcept>

namespace py = pybind11;

namespace pybind11 { namespace detail {

// torch.autograd.Variable <-> at::Tensor conversions (without unwrapping)
template <>
struct type_caster<at::Tensor> {
 public:
  PYBIND11_TYPE_CASTER(at::Tensor, _("at::Tensor"));

  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (THPVariable_Check(obj)) {
      value = reinterpret_cast<THPVariable*>(obj)->cdata;
      return true;
    }
    return false;
  }

  static handle
  cast(at::Tensor src, return_value_policy /* policy */, handle /* parent */) {
    if (!torch::autograd::is_variable(src)) {
      throw std::runtime_error(
          "Expected tensor's dynamic type to be Variable, not Tensor");
    }
    return handle(THPVariable_Wrap(torch::autograd::Variable(src)));
  }
};

template<> struct type_caster<torch::autograd::Variable> {
public:
  PYBIND11_TYPE_CASTER(torch::autograd::Variable, _("torch::autograd::Variable"));
  bool load(handle src, bool) {
    PyObject *source = src.ptr();
    if (THPVariable_Check(source)) {
      value = ((THPVariable*)source)->cdata;
      return true;
    } else {
      return false;
    }
  }
  static handle cast(torch::autograd::Variable src, return_value_policy /* policy */, handle /* parent */) {
    return handle(THPVariable_Wrap(src));
  }
};

}} // namespace pybind11::detail
