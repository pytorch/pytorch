// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include <functorch/csrc/dim/dim.h>

namespace at {
namespace functorch {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // initialize first-class dims and install it as a submodule on _C
  auto dim = Dim_init();
  if (!dim) {
    throw py::error_already_set();
  }
  py::setattr(m, "dim", py::reinterpret_steal<py::object>(dim));
}

}}
