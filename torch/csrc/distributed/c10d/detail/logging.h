// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <utility>

#include <fmt/format.h>

#include <c10/util/Logging.h>

namespace c10d {
namespace detail {

template <typename... Args>
inline void logInfo(fmt::string_view msg, Args&&... args) {
  if (FLAGS_caffe2_log_level <= 0) {
    LOG(INFO) << fmt::format(msg, std::forward<Args>(args)...);
  }
}

template <typename... Args>
inline void logWarning(fmt::string_view msg, Args&&... args) {
  if (FLAGS_caffe2_log_level <= 1) {
    LOG(WARNING) << fmt::format(msg, std::forward<Args>(args)...);
  }
}

template <typename... Args>
inline void logError(fmt::string_view msg, Args&&... args) {
  if (FLAGS_caffe2_log_level <= 2) {
    LOG(ERROR) << fmt::format(msg, std::forward<Args>(args)...);
  }
}

} // namespace detail
} // namespace c10d
