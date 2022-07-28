// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <torch/csrc/utils/pybind.h>

namespace at {
namespace functorch {

/// Initialize python bindings for kernel compilation cache.
void initCompileCacheBindings(PyObject *module);

} // namespace functorch
} // namespace at
