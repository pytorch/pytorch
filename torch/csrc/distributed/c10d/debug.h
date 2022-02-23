// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <c10/macros/Macros.h>

namespace c10d {

enum class DebugLevel {
  Off,
  Info,
  Detail
};

TORCH_API void setDebugLevel(DebugLevel level);

// Sets the debug level based on the value of the `TORCH_DISTRIBUTED_DEBUG`
// environment variable.
TORCH_API void setDebugLevelFromEnvironment();

TORCH_API DebugLevel debug_level() noexcept;

} // namespace c10d
