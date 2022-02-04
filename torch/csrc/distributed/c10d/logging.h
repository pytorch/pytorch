// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <fmt/format.h>

#include <c10/util/Logging.h>

namespace c10d {
namespace detail {
template <typename... T>
std::string log_vformat(fmt::string_view fmt, T&&... args) {
  return fmt::vformat(fmt, fmt::make_format_args(args...));
}
}  // namespace detail
}  // namespace c10d

#define C10D_ERROR(...)\
    LOG_IF(ERROR,   FLAGS_caffe2_log_level <= 2) << c10d::detail::log_vformat(__VA_ARGS__)

#define C10D_WARNING(...)\
    LOG_IF(WARNING, FLAGS_caffe2_log_level <= 1) << c10d::detail::log_vformat(__VA_ARGS__)

#define C10D_INFO(...)\
    LOG_IF(INFO,    FLAGS_caffe2_log_level <= 0) << c10d::detail::log_vformat(__VA_ARGS__)
