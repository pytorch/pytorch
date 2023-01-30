// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/csrc/distributed/c10d/logging.h>

#include <torch/csrc/distributed/c10d/debug.h>

namespace c10d {
namespace detail {

bool isLogLevelEnabled(LogLevel level) noexcept {
  // c10 logger does not support debug and trace levels. In order to map higher
  // levels we adjust our ordinal value.
  int level_int = static_cast<int>(level) - 2;

  if (level_int >= 0) {
    return FLAGS_caffe2_log_level <= level_int;
  }

  // Debug and trace levels are only enabled when c10 log level is set to INFO.
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
