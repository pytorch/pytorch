// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>

namespace c10d {

enum class DebugLevel {
  Off,
  Info,
  Detail
};

TORCH_API void setDebugLevel(at::optional<DebugLevel> opt_level = {}, bool force = false);

TORCH_API DebugLevel debug_level() noexcept;

} // namespace c10d
