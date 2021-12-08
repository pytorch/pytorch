// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/csrc/MetaInit.h>

#include <ATen/MetaInit.h>
#include <ATen/Tensor.h>

#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace torch {

void initMetaInitFunctions(py::module& m) {
  m.def("_enable_meta_init", at::enableMetaInit);

  m.def("_is_meta_init_enabled", at::isMetaInitEnabled);

  m.def("_materialize_tensor", at::materializeTensor);

  m.def("_clear_meta_init_cache", at::clearMetaInitCache);
}

} // namespace torch
