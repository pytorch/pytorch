// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <c10d/logging.h>

#include <c10d/debug.h>

namespace c10d {
namespace detail {

bool isLogLevelEnabled(LogLevel level) noexcept {
  // c10d logger does not support debug and verbose levels. In order to map the
  // remaining levels we adjust our ordinal value.
  int level_int = static_cast<int>(level) - 2;

  if (level_int >= 0) {
    return FLAGS_caffe2_log_level <= level_int;
  }

  // Debug and verbose levels are only enabled when c10 log level is INFO.
  if (FLAGS_caffe2_log_level != 0) {
    return false;
  }

  if (level_int == -1) {
    return debug_level() != DebugLevel::Off;
  }
  if (level_int == -2) {
    return debug_level() == DebugLevel::Detail;
  }

  return false;
}

} // namespace detail
} // namespace c10d
