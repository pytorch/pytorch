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
