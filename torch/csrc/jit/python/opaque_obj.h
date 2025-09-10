#pragma once

#include <string>

#include <c10/macros/Macros.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/custom_class.h>

namespace torch::jit {
struct TORCH_API OpaqueObject : public CustomClassHolder {
  OpaqueObject(py::handle payload) : payload_(payload) {}

  void setPayload(py::handle payload) {
    payload_ = payload;
  }

  py::handle getPayload() {
    return payload_;
  }

  py::handle payload_;
};

static auto register_opaque_obj_class =
    torch::class_<OpaqueObject>("aten", "OpaqueObject");

} // namespace torch::jit
