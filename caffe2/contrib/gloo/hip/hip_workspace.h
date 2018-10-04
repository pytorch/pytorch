/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "hip.h"

namespace gloo {

// HIP workspaces
//
// Algorithms take a workspace template argument and if it uses the
// HipDeviceWorkspace can be used with a GPUDirect capable transport.

template <typename T>
class HipHostWorkspace {
 public:
  using Pointer = HipHostPointer<T>;
};

template <typename T>
class HipDeviceWorkspace {
 public:
  using Pointer = HipDevicePointer<T>;
};

} // namespace gloo
