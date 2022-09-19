// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace functorch {

// CompileCache is the compilation cache used by the AOTAutograd frontend.
// We're planning on deleting this in favor of torchdynamo's caching mechanism
// (CompilerCache predates torchdynamo).

/// Initialize python bindings for kernel compilation cache.
TORCH_API void initCompileCacheBindings(PyObject* module);

} // namespace functorch
} // namespace torch
