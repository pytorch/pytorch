// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <mutex>
#include <string>

#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <fmt/format.h>

namespace c10d {
namespace detail {

enum class LogLevel { Trace, Debug, Info, Warning, Error };

TORCH_API bool isLogLevelEnabled(LogLevel level) noexcept;

template <typename... T>
std::string formatLogMessage(fmt::string_view fmt, T&&... args) {
  return fmt::vformat(fmt, fmt::make_format_args(args...));
}

// logWithLogging is a wrapper around std::unique_lock<std::timed_mutex>
// that automatically logs if the lock cannot be acquired within a given
// timeout.
TORCH_API void lockWithLogging(
    std::unique_lock<std::timed_mutex>& lock,
    std::chrono::milliseconds log_interval,
    c10::string_view desc,
    c10::string_view file,
    int line);

} // namespace detail
} // namespace c10d

#define C10D_ERROR(...)                                                      \
  LOG_IF(                                                                    \
      ERROR, c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Error)) \
      << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_WARNING(...)                                               \
  LOG_IF(                                                               \
      WARNING,                                                          \
      c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Warning)) \
      << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_INFO(...)                                                        \
  LOG_IF(INFO, c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Info)) \
      << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_DEBUG(...)                                                        \
  LOG_IF(INFO, c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Debug)) \
      << "[c10d - debug] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_TRACE(...)                                                        \
  LOG_IF(INFO, c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Trace)) \
      << "[c10d - trace] " << c10d::detail::formatLogMessage(__VA_ARGS__)

// TODO: use std::source_location() when we can use C++20
#define C10D_LOCK_GUARD(name, mutex)                               \
  std::unique_lock<std::timed_mutex> name{mutex, std::defer_lock}; \
  ::c10d::detail::lockWithLogging(                                 \
      name, std::chrono::seconds(30), #mutex, __FILE__, __LINE__)
