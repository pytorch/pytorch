#pragma once

#include <ATen/ATenGeneral.h> // for AT_API

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace at {
namespace detail {
struct SourceLocation {
  const char* function;
  const char* file;
  uint32_t line;
};
} // namespace detail

template <typename... FormatArgs>
[[noreturn]] AT_API void error(
    detail::SourceLocation source_location,
    const char* format_string,
    FormatArgs&&... format_args) {
  static const size_t kMaximumErrorMessageLength = 4096;

  std::string format_string_with_source_location(format_string);
  format_string_with_source_location += " (%s at %s:%d)";

  char buffer[kMaximumErrorMessageLength];
  snprintf(
      buffer,
      sizeof(buffer),
      format_string_with_source_location.c_str(),
      format_args...,
      source_location.function,
      source_location.file,
      source_location.line);

  throw std::runtime_error(buffer);
}
} // namespace at

#define AT_ERROR(...) at::error({__func__, __FILE__, __LINE__}, __VA_ARGS__)
