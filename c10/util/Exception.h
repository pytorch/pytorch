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
/// primarily be used with the `C10_ERROR` macro.
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
      const int line,
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

// A utility function to return an exception std::string by prepending its
// exception type before its what() content
C10_API std::string GetExceptionString(const std::exception& e);

} // namespace c10

// TODO: variants that print the expression tested and thus don't require
// strings
// TODO: CAFFE_ENFORCE_WITH_CALLER style macro
// TODO: consolidate the enforce and assert messages. Assert is a bit confusing
// as c++ assert quits, while this throws.

#define C10_ERROR(...) \
  throw ::c10::Error({__func__, __FILE__, __LINE__}, ::c10::str(__VA_ARGS__))
#define AT_ERROR C10_ERROR

#define C10_WARN(...) \
  ::c10::Warning::warn({__func__, __FILE__, __LINE__}, ::c10::str(__VA_ARGS__))
#define AT_WARN C10_WARN

// C10_ASSERT is used to guard programming errors, not user errors. If it is
// triggered, it should mean that a pytorch bug is found. For user error, use
// C10_ENFORCE.
// Note: strictly, this is wrong, because if we identify a programming error,
// we should exit early and not continue execution. This is further complicated
// by the fact that we are using the same error type (c10::Error) to notify
// the user, and the only way to distinguish it is the string telling users to
// report a bug. For errors that are not recoverable (like leading to a
// corrupted internal state), consider directly CHECK instead of asserting.
// TODO: move to CHECK explicitly in such cases.
#define C10_ASSERT(cond, ...)                \
  if (!(cond)) {                             \
    C10_ERROR(                               \
        #cond " ASSERT FAILED at ",          \
        __FILE__,                            \
        ":",                                 \
        __LINE__,                            \
        ", please report a bug to PyTorch.", \
        ##__VA_ARGS__);                      \
  }
#define AT_ASSERT C10_ASSERT
#define AT_ASSERTM C10_ASSERT

#define C10_ENFORCE(cond, ...) \
  if (!(cond)) {               \
    C10_ERROR(__VA_ARGS__);    \
  }
#define AT_CHECK C10_ENFORCE

#endif // C10_UTIL_EXCEPTION_H_
