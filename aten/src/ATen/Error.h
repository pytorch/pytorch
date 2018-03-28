#pragma once

#include <ATen/ATenGeneral.h> // for AT_API

#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

namespace at {
namespace detail {
/// A printf wrapper that returns an std::string.
template <typename... FormatArgs>
std::string format(const char* format_string, FormatArgs&&... format_args) {
  static constexpr size_t kMaximumStringLength = 4096;
  char buffer[kMaximumStringLength];
  snprintf(buffer, sizeof(buffer), format_string, format_args...);
  return buffer;
}

/// Represents a location in source code (for debugging).
struct SourceLocation {
  std::string toString() const {
    return format("%s at %s:%d", function, file, line);
  }

  const char* function;
  const char* file;
  uint32_t line;
};
} // namespace detail

/// The primary ATen error class.
/// Provides a complete error message with source location information via
/// `what()`, and a more concise message via `what_without_location()`. Should
/// primarily be used with the `AT_ERROR` macro.
struct AT_API Error : public std::exception {
  template <typename... FormatArgs>
  Error(
      detail::SourceLocation source_location,
      const char* format_string,
      FormatArgs&&... format_args)
      : what_without_location_(detail::format(
            format_string,
            std::forward<FormatArgs>(format_args)...)),
        what_(
            what_without_location_ + " (" + source_location.toString() + ")") {}

  /// Returns the complete error message including the source location.
  const char* what() const noexcept override {
    return what_.c_str();
  }

  /// Returns only the error message string, without source location.
  const char* what_without_location() const noexcept {
    return what_without_location_.c_str();
  }

 private:
  std::string what_without_location_;
  std::string what_;
};
} // namespace at

#define AT_ERROR(...) \
  throw at::Error({__func__, __FILE__, __LINE__}, __VA_ARGS__)

#define AT_ASSERT(cond, ...) \
  if (!(cond)) {             \
    AT_ERROR(__VA_ARGS__);   \
  }
