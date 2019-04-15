#ifndef C10_UTIL_EXCEPTION_H_
#define C10_UTIL_EXCEPTION_H_

#include "c10/macros/Macros.h"
#include "c10/util/StringUtil.h"

#include <cstddef>
#include <exception>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__
#endif

namespace c10 {

/// The primary ATen error class.
/// Provides a complete error message with source location information via
/// `what()`, and a more concise message via `what_without_backtrace()`. Should
/// primarily be used with the `AT_ERROR` macro.
///
/// NB: c10::Error is handled specially by the default torch to suppress the
/// backtrace, see torch/csrc/Exceptions.h
class C10_API Error : public std::exception {
  std::vector<std::string> msg_stack_;
  std::string backtrace_;

  // These two are derived fields from msg_stack_ and backtrace_, but we need
  // fields for the strings so that we can return a const char* (as the
  // signature of std::exception requires).
  std::string msg_;
  std::string msg_without_backtrace_;

  // This is a little debugging trick: you can stash a relevant pointer
  // in caller, and then when you catch the exception, you can compare
  // against pointers you have on hand to get more information about
  // where the exception came from.  In Caffe2, this is used to figure
  // out which operator raised an exception.
  const void* caller_;

 public:
  Error(
      const std::string& msg,
      const std::string& backtrace,
      const void* caller = nullptr);
  Error(SourceLocation source_location, const std::string& msg);
  Error(
      const char* file,
      const uint32_t line,
      const char* condition,
      const std::string& msg,
      const std::string& backtrace,
      const void* caller = nullptr);

  void AppendMessage(const std::string& msg);

  // Compute the full message from msg_ and msg_without_backtrace_
  // TODO: Maybe this should be private
  std::string msg() const;
  std::string msg_without_backtrace() const;

  const std::vector<std::string>& msg_stack() const {
    return msg_stack_;
  }

  /// Returns the complete error message, including the source location.
  const char* what() const noexcept override {
    return msg_.c_str();
  }

  const void* caller() const noexcept {
    return caller_;
  }

  /// Returns only the error message string, without source location.
  const char* what_without_backtrace() const noexcept {
    return msg_without_backtrace_.c_str();
  }
};

class C10_API Warning {
  using handler_t =
      void (*)(const SourceLocation& source_location, const char* msg);

 public:
  /// Issue a warning with a given message. Dispatched to the current
  /// warning handler.
  static void warn(SourceLocation source_location, std::string msg);
  /// Sets the global warning handler. This is not thread-safe, so it should
  /// generally be called once during initialization.
  static void set_warning_handler(handler_t handler);
  /// The default warning handler. Prints the message to stderr.
  static void print_warning(
      const SourceLocation& source_location,
      const char* msg);

 private:
  static handler_t warning_handler_;
};

// Used in ATen for out-of-bound indices that can reasonably only be detected
// lazily inside a kernel (See: advanced indexing).
class C10_API IndexError : public Error {
  using Error::Error;
};


// A utility function to return an exception std::string by prepending its
// exception type before its what() content
C10_API std::string GetExceptionString(const std::exception& e);

} // namespace c10

// TODO: variants that print the expression tested and thus don't require
// strings
// TODO: CAFFE_ENFORCE_WITH_CALLER style macro

// TODO: move AT_ERROR to C10_ERROR
// TODO: consolidate the enforce and assert messages. Assert is a bit confusing
// as c++ assert quits, while this throws.
// TODO: merge AT_CHECK with AT_ASSERTM. CHECK in fbcode means strict failure if
// not met.

// In the debug build With MSVC, __LINE__ might be of long type (a.k.a int32_t),
// which is different from the definition of `SourceLocation` that requires
// unsigned int (a.k.a uint32_t) and may cause a compile error with the message:
// error C2397: conversion from 'long' to 'uint32_t' requires a narrowing conversion
// Here the static cast is used to pass the build.

#define AT_ERROR(...) \
  throw ::c10::Error({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, ::c10::str(__VA_ARGS__))

#define AT_INDEX_ERROR(...) \
  throw ::c10::IndexError({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, ::c10::str(__VA_ARGS__))

#define AT_WARN(...) \
  ::c10::Warning::warn({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, ::c10::str(__VA_ARGS__))

#define AT_ASSERT(cond)                       \
  if (!(cond)) {                              \
    AT_ERROR(                                 \
        #cond " ASSERT FAILED at ",           \
        __FILE__,                             \
        ":",                                  \
        __LINE__,                             \
        ", please report a bug to PyTorch."); \
  }

#define AT_ASSERTM(cond, ...)                 \
  if (!(cond)) {                              \
    AT_ERROR(::c10::str(                      \
        #cond,                                \
        " ASSERT FAILED at ",                 \
        __FILE__,                             \
        ":",                                  \
        __LINE__,                             \
        ", please report a bug to PyTorch. ", \
        __VA_ARGS__));                        \
  }

#define AT_CHECK(cond, ...)            \
  if (!(cond)) {                       \
    AT_ERROR(::c10::str(__VA_ARGS__)); \
  }

#endif // C10_UTIL_EXCEPTION_H_
