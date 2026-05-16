// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <atomic>
#include <string>

#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <fmt/format.h>

namespace c10d::detail {

enum class LogLevel { Trace, Debug, Info, Warning, Error };

TORCH_API bool isLogLevelEnabled(LogLevel level) noexcept;

template <typename... T>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
std::string formatLogMessage(fmt::string_view fmt, T&&... args) {
  return fmt::vformat(fmt, fmt::make_format_args(args...));
}

} // namespace c10d::detail

#define C10D_ERROR(...)                                               \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Error)) \
  LOG(ERROR) << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_WARNING(...)                                               \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Warning)) \
  LOG(WARNING) << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_INFO(...)                                               \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Info)) \
  LOG(INFO) << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_DEBUG(...)                                               \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Debug)) \
  LOG(INFO) << "[c10d - debug] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_TRACE(...)                                               \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Trace)) \
  LOG(INFO) << "[c10d - trace] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_WARNING_EVERY_NTH(n, ...)                                  \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Warning)) \
  C10_LOG_EVERY_NTH(WARNING, n)                                         \
      << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_INFO_EVERY_NTH(n, ...)                                  \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Info)) \
  C10_LOG_EVERY_NTH(INFO, n)                                         \
      << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_DEBUG_EVERY_NTH(n, ...)                                  \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Debug)) \
  C10_LOG_EVERY_NTH(INFO, n)                                          \
      << "[c10d - debug] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_TRACE_EVERY_NTH(n, ...)                                  \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Trace)) \
  C10_LOG_EVERY_NTH(INFO, n)                                          \
      << "[c10d - trace] " << c10d::detail::formatLogMessage(__VA_ARGS__)

// Logs at WARNING on calls N, 2N, 3N, ... and at DEBUG otherwise.
// Useful for retry loops where each failure is worth recording but only
// every Nth one needs to surface as a warning.
#define C10D_WARNING_EVERY_N_ELSE_DEBUG(n, ...)                    \
  do {                                                             \
    static std::atomic<size_t> _c10d_counter{0};                   \
    size_t _c10d_c =                                               \
        _c10d_counter.fetch_add(1, std::memory_order_relaxed) + 1; \
    if (_c10d_c % (n) == 0) {                                      \
      C10D_WARNING(__VA_ARGS__);                                   \
    } else {                                                       \
      C10D_DEBUG(__VA_ARGS__);                                     \
    }                                                              \
  } while (false)
