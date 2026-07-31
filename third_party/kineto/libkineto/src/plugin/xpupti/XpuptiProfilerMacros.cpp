/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiProfilerMacros.h"

#include <stdexcept>
#include <string_view>

#include <fmt/format.h>

namespace KINETO_NAMESPACE {

namespace {
[[noreturn]] void throwXpuRuntimeError(
    pti_result errCode,
    std::string_view message,
    std::source_location source_location) {
  const std::string function_location = fmt::format(
      "function {} located: {}:{}",
      source_location.function_name(),
      source_location.file_name(),
      source_location.line());
  const std::string error_code = fmt::format(
      "Error code: {} ({})",
      static_cast<int>(errCode),
      ptiResultTypeToString(errCode));
  const std::string error = fmt::format(
      "Kineto Profiler on XPU got error from {}. {}.",
      function_location,
      error_code);

  if (message.empty())
    throw std::runtime_error(error);
  else {
    const std::string error_with_message =
        fmt::format("{}. Message: {}", error, message);
    throw std::runtime_error(error_with_message);
  }
}
} // namespace

void XPUPTI_CALL(
    pti_result errCode,
    std::string_view message,
    std::source_location source_location) {
  if (errCode != PTI_SUCCESS)
    throwXpuRuntimeError(errCode, message, source_location);
}

} // namespace KINETO_NAMESPACE
