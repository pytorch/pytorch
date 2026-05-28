// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/comms/gloo/TorchCommGloo.hpp>

namespace py = pybind11;
using namespace torch::comms;

void init_comms_gloo_bindings(py::module_& m) {
  m.doc() = "Gloo specific python bindings for TorchComm";

  py::class_<TorchCommGloo, TorchCommBackend, std::shared_ptr<TorchCommGloo>>(
      m, "TorchCommGloo");
}
