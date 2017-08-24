#pragma once

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "torch/csrc/DynamicTypes.h"

namespace py = pybind11;

namespace pybind11 { namespace detail {

// handle Tensor <-> at::Tensor conversions
template <> struct type_caster<at::Tensor> {
public:
  PYBIND11_TYPE_CASTER(at::Tensor, _("at::Tensor"));

  bool load(handle src, bool) {
    try {
      value = torch::createTensor(src.ptr());
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

