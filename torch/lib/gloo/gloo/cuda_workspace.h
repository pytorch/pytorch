/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/cuda.h"

namespace gloo {

// CUDA workspaces
//
// Algorithms take a workspace template argument and if it uses the
// CudaDeviceWorkspace can be used with a GPUDirect capable transport.

template <typename T>
class CudaHostWorkspace {
 public:
  using Pointer = CudaHostPointer<T>;
};

template <typename T>
class CudaDeviceWorkspace {
 public:
  using Pointer = CudaDevicePointer<T>;
};

} // namespace gloo
