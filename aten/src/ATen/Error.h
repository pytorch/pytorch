#pragma once

#include <ATen/ATenGeneral.h> // for AT_API
#include <ATen/optional.h>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <stdarg.h>

#if !defined(_WIN32)
#include <cxxabi.h>
#include <execinfo.h>
#endif // !defined(_WIN32)

#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__
#endif

// This is a mash-up of ATen/Error.h and caffe2/core/logging.h

namespace at {

namespace detail {

// Utility to demangle a function name
std::string demangle(const char* name);

/**
 * Returns the printable name of the type.
 *
 * Works for all types, not only the ones registered with CAFFE_KNOWN_TYPE
 */
template <typename T>
static const char* demangle_type() {
#ifdef __GXX_RTTI
  static const std::string name = demangle(typeid(T).name());
  return name.c_str();
#else // __GXX_RTTI
  return "(RTTI disabled, cannot show name)";
#endif // __GXX_RTTI
}

inline void _str(std::stringstream& /*ss*/) {}

template <typename T>
inline void _str(std::stringstream& ss, const T& t) {
  ss << t;
}

template <typename T, typename... Args>
inline void
_str(std::stringstream& ss, const T& t, const Args&... args) {
  _str(ss, t);
  _str(ss, args...);
}

} // namespace detail

// Convert a list of string-like arguments into a single string.
template <typename... Args>
inline std::string str(const Args&... args) {
  std::stringstream ss;
  detail::_str(ss, args...);
  return ss.str();
}

// Specializations for already-a-string types.
template <>
inline std::string str(const std::string& str) {
  return str;
}
inline std::string str(const char* c_str) {
  return c_str;
}

/// Represents a location in source code (for debugging).
struct SourceLocation {
  const char* function;
  const char* file;
  uint32_t line;
};

inline std::ostream& operator<<(std::ostream& out, const SourceLocation& loc) {
  out << loc.function << " at " << loc.file << ":" << loc.line;
  return out;
}

std::string get_backtrace(
    size_t frames_to_skip = 0,
    size_t maximum_number_of_frames = 64);

/// The primary ATen error class.
/// Provides a complete error message with source location information via
/// `what()`, and a more concise message via `what_without_backtrace()`. Should
/// primarily be used with the `AT_ERROR` macro.
class AT_API Error : public std::exception {
  std::string what_without_backtrace_;
  std::string what_;

public:
  Error(SourceLocation source_location, std::string err)
    : what_without_backtrace_(err)
    , what_(str("\n", err, " (", source_location, ")\n", get_backtrace(/*frames_to_skip=*/2)))
  {}

  /// Returns the complete error message, including the source location.
  inline const char* what() const noexcept override {
    return what_.c_str();
  }

  /// Returns only the error message string, without source location.
  inline const char* what_without_backtrace() const noexcept {
    return what_without_backtrace_.c_str();
  }
};

} // namespace at

// TODO: variants that print the expression tested and thus don't require strings
// TODO: CAFFE_ENFORCE_WITH_CALLER style macro

#define AT_ERROR(...) \
  throw at::Error({__func__, __FILE__, __LINE__}, at::str(__VA_ARGS__))

#define AT_ASSERT(cond, ...) \
  if (!(cond)) {             \
    AT_ERROR(at::str(#cond, " ", __VA_ARGS__));   \
  }

#define AT_CHECK(cond, ...) \
  if (!(cond)) {             \
    AT_ERROR(at::str(__VA_ARGS__));   \
  }
