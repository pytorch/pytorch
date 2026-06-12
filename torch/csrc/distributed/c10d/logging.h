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

// _EVERY_N variants wrap c10's C10_LOG_EVERY_N. Logs on calls
// 1, N+1, 2N+1, ... On non-glog builds throttling is not available
// and these fall back to logging every call.
#define C10D_WARNING_EVERY_N(n, ...)                                    \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Warning)) \
  C10_LOG_EVERY_N(WARNING, n)                                           \
      << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_INFO_EVERY_N(n, ...)                                    \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Info)) \
  C10_LOG_EVERY_N(INFO, n) << "[c10d] "                              \
                           << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_DEBUG_EVERY_N(n, ...)                                    \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Debug)) \
  C10_LOG_EVERY_N(INFO, n) << "[c10d - debug] "                       \
                           << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_TRACE_EVERY_N(n, ...)                                    \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Trace)) \
  C10_LOG_EVERY_N(INFO, n) << "[c10d - trace] "                       \
                           << c10d::detail::formatLogMessage(__VA_ARGS__)

// _IF_EVERY_N variants wrap c10's C10_LOG_IF_EVERY_N. Logs on calls
// where the condition is true, throttled to the 1st, N+1th, 2N+1th ...
// such call. On non-glog builds throttling is not available and these
// fall back to LOG_IF(severity, condition) (every truthy call logs).
#define C10D_WARNING_IF_EVERY_N(condition, n, ...)                      \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Warning)) \
  C10_LOG_IF_EVERY_N(WARNING, condition, n)                             \
      << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_INFO_IF_EVERY_N(condition, n, ...)                      \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Info)) \
  C10_LOG_IF_EVERY_N(INFO, condition, n)                             \
      << "[c10d] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_DEBUG_IF_EVERY_N(condition, n, ...)                      \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Debug)) \
  C10_LOG_IF_EVERY_N(INFO, condition, n)                              \
      << "[c10d - debug] " << c10d::detail::formatLogMessage(__VA_ARGS__)

#define C10D_TRACE_IF_EVERY_N(condition, n, ...)                      \
  if (c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Trace)) \
  C10_LOG_IF_EVERY_N(INFO, condition, n)                              \
      << "[c10d - trace] " << c10d::detail::formatLogMessage(__VA_ARGS__)

// _EVERY_NTH variants wrap c10's C10_LOG_EVERY_NTH (counter-based). Logs on
// calls N, 2N, 3N, ... (the first N-1 calls are not logged).
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
#define C10D_WARNING_EVERY_NTH_ELSE_DEBUG(n, ...)                  \
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
