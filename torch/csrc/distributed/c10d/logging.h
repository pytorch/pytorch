// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>

#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <fmt/format.h>

namespace c10d {
namespace detail {

enum class LogLevel {
  Trace,
  Debug,
  Info,
  Warning,
  Error
};

TORCH_API bool isLogLevelEnabled(LogLevel level) noexcept;

template <typename... T>
std::string formatLogMessage(fmt::string_view fmt, T&&... args) {
  return fmt::vformat(fmt, fmt::make_format_args(args...));
}

} // namespace detail
} // namespace c10d

#define C10D_ERROR(...)\
    LOG_IF(ERROR,   c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Error))\
        << "[c10d] "         << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_WARNING(...)\
    LOG_IF(WARNING, c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Warning))\
        << "[c10d] "         << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_INFO(...)\
    LOG_IF(INFO,    c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Info))\
        << "[c10d] "         << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_DEBUG(...)\
    LOG_IF(INFO,    c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Debug))\
        << "[c10d - debug] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_TRACE(...)\
    LOG_IF(INFO,    c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Trace))\
        << "[c10d - trace] " << c10d::detail::formatLogMessage(__VA_ARGS__)
