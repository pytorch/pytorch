#pragma once

#include <string>

#include <c10/macros/Macros.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/custom_class.h>

namespace torch::jit {
struct OpaqueObject : public CustomClassHolder {
  OpaqueObject(py::object payload) : payload_(payload) {}

  void setPayload(py::object payload) {
    payload_ = payload;
  }

  py::object getPayload() {
    return payload_;
  }

  py::object payload_;
};

static auto register_opaque_obj_class =
    torch::class_<OpaqueObject>("aten", "OpaqueObject");

} // namespace torch::jit
