#pragma once

#include <torch/csrc/python_headers.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_cpp_function.h>

namespace py = pybind11;

namespace pybind11 { namespace detail {

// handle Python <-> torch::autograd::Function conversions
template <> struct type_caster<std::shared_ptr<torch::autograd::Function>> {
public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<torch::autograd::Function>, _("std::shared_ptr<torch::autograd::Function>"));

  bool load(handle src, bool) {
    if (!THPFunction_Check(src.ptr())) return false;
    value = THPFunction_asFunction((THPFunction*)src.ptr());
    return true;
  }
  static handle cast(std::shared_ptr<torch::autograd::Function> src, return_value_policy /* policy */, handle /* parent */) {
    auto fn = functionToPyObject(std::move(src));
    return handle(fn);
  }
};


}} // namespace pybind11::detail
