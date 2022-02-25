#ifndef MULTIPY_UTIL_EXCEPTION_H_
#define MULTIPY_UTIL_EXCEPTION_H_

#include <cstddef>
#include <exception>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__
#endif

#define MULTIPY_STRINGIZE(x) #x

struct SourceLocation {
  const std::string& function;
  const std::string& file;
  uint32_t line;
};

namespace multipyError {
  /// The primary MultiPy error class.
  /// Provides a complete error message with source location information via
  /// `what()`, and a more concise message via `what_without_backtrace()`.
  /// Don't throw this directly; use MULTIPLY_CHECK/MULTIPY_INTERNAL_ASSERT instead.
  ///
  /// NB: multipyError::Error is handled specially by the default torch to suppress the
  /// backtrace, see torch/csrc/Exceptions.h
  // Caffe2-style error message
  class Error : public std::exception {
  // The actual error message.
  std::string msg_;

  // Context for the message (in order of decreasing specificity).  Context will
  // be automatically formatted appropriately, so it is not necessary to add
  // extra leading/trailing newlines to strings inside this vector
  std::vector<std::string> context_;

  // The C++ backtrace at the point when this exception was raised.  This
  // may be empty if there is no valid backtrace.  (We don't use optional
  // here to reduce the dependencies this file has.)
  std::string backtrace_;

  // These two are derived fields from msg_stack_ and backtrace_, but we need
  // fields for the strings so that we can return a const std::string& (as the
  // signature of std::exception requires).  Currently, the invariant
  // is that these fields are ALWAYS populated consistently with respect
  // to msg_stack_ and backtrace_.
  std::string what_;
  std::string what_without_backtrace_;

  // This is a little debugging trick: you can stash a relevant pointer
  // in caller, and then when you catch the exception, you can compare
  // against pointers you have on hand to get more information about
  // where the exception came from.  In Caffe2, this is used to figure
  // out which operator raised an exception.
  const void* caller_;
  std::string get_backtrace(
    size_t frames_to_skip = 0,
    size_t maximum_number_of_frames = 64,
    bool skip_python_frames = true);
  public:
  // PyTorch-style Error constructor.

  Error(SourceLocation source_location, std::string msg);

  Error(
      const std::string& file,
      const uint32_t line,
      const std::string& condition,
      const std::string& msg,
      const std::string& backtrace,
      const void* caller = nullptr);

  // Base constructor
  Error(std::string msg, std::string backtrace, const void* caller = nullptr);

  // Add some new context to the message stack.  The last added context
  // will be formatted at the end of the context list upon printing.
  // WARNING: This method is O(n) in the size of the stack, so don't go
  // wild adding a ridiculous amount of context to error messages.
  void add_context(std::string msg);

  const std::string& msg() const {
    return msg_;
  }

  const std::vector<std::string>& context() const {
    return context_;
  }

  const std::string& backtrace() const {
    return backtrace_;
  }

  const void* caller() const noexcept {
    return caller_;
  }

  /// Returns only the error message string, without source location.
  /// The returned pointer is invalidated if you call add_context() on
  /// this object.
  const std::string& what_without_backtrace() const noexcept {
    return what_without_backtrace_.c_str();
  }

 private:
  void refresh_what();
  std::string compute_what(bool include_backtrace) const;
};


} // namespace multipyError

// A utility macro to provide assert()-like functionality; that is, enforcement
// of internal invariants in code.  It supports an arbitrary number of extra
// arguments (evaluated only on failure), which will be printed in the assert
// failure message using operator<< (this is useful to print some variables
// which may be useful for debugging.)
//
// Usage:
//    MULTIPY_INTERNAL_ASSERT(should_be_true);
//    MULTIPY_INTERNAL_ASSERT(x == 0, "x = ", x);
//
// Assuming no bugs in PyTorch, the conditions tested by this macro should
// always be true; e.g., it should be possible to disable all of these
// conditions without changing observable user behavior.  If you would like to
// do error reporting for user input, please use MULTIPLY_CHECK instead.
//
// NOTE: It is SAFE to use this macro in production code; on failure, this
// simply raises an exception, it does NOT unceremoniously quit the process
// (unlike assert()).
//
#ifdef STRIP_ERROR_MESSAGES
#define MULTIPY_INTERNAL_ASSERT(cond, ...)                            \
  ::multipyError::detail::multipyCheckFail(                                  \
      __func__,                                                   \
      __FILE__,                                                   \
      static_cast<uint32_t>(__LINE__),                            \
      #cond "INTERNAL ASSERT FAILED at" MULTIPY_STRINGIZE(__FILE__));
#else
// It would be nice if we could build a combined string literal out of
// the MULTIPY_INTERNAL_ASSERT prefix and a user-provided string literal
// as the first argument, but there doesn't seem to be any good way to
// do that while still supporting having a first argument that isn't a
// string literal.
#define MULTIPY_INTERNAL_ASSERT(cond, ...)                                        \
  ::multipyError::detail::multipyInternalAssertFail(                                     \
      __func__,                                                               \
      __FILE__,                                                               \
      static_cast<uint32_t>(__LINE__),                                        \
      #cond                                                                   \
      "INTERNAL ASSERT FAILED at " MULTIPY_STRINGIZE(__FILE__) ":" MULTIPY_STRINGIZE( \
          __LINE__) ", please report a bug to PyTorch/MultiPy team. ",                     \
      MULTIPY_STRINGIZE(__VA_ARGS__));
#endif // STRIP_ERROR_MESSAGES

#ifdef STRIP_ERROR_MESSAGES
#define MULTIPLY_CHECK_MSG(cond, type, ...) \
  (#cond #type " CHECK FAILED at " MULTIPY_STRINGIZE(__FILE__))
#else
namespace multipyError {
namespace detail {
std::string StripBasename(const std::string& full_path);
template <typename... Args>
decltype(auto) torchCheckMsgImpl(const std::string& msg, const Args&... args) {
  return MULTIPY_STRINGIZE(args);
}
inline const std::string& torchCheckMsgImpl(const std::string& msg) {
  return msg;
}
// If there is just 1 user-provided C-string argument, use it.
inline const std::string& torchCheckMsgImpl(
    const std::string& msg,
    const std::string& args) {
  return args;
}
} // namespace detail
} // namespace multipyError

#define MULTIPLY_CHECK_MSG(cond, type, ...)                   \
  (::multipyError::detail::torchCheckMsgImpl(                       \
      "Expected " #cond                                    \
      " to be true, but got false.  "                      \
      "(Could this error message be improved?  If so, "    \
      "please report an enhancement request to PyTorch.)", \
      ##__VA_ARGS__))
#define MULTIPLY_CHECK_WITH_MSG(error_t, cond, type, ...)                  \
  C10_THROW_ERROR(error_t, MULTIPLY_CHECK_MSG(cond, type, __VA_ARGS__));
#endif // STRIP_ERROR_MESSAGES

namespace multipyError {
namespace detail {

[[noreturn]] inline void multipyCheckFail(
    const std::string& func,
    const std::string& file,
    uint32_t line,
    const std::string& msg);

[[noreturn]] void multipyInternalAssertFail(
    const std::string& func,
    const std::string& file,
    uint32_t line,
    const std::string& condMsg,
    const std::string& userMsg);

} // namespace detail
} // namespace multipyError

#ifdef STRIP_ERROR_MESSAGES
#define MULTIPLY_CHECK(cond, ...)                   \
  ::multipyError::detail::multipyCheckFail(               \
      __func__,                                \
      __FILE__,                                \
      static_cast<uint32_t>(__LINE__),         \
      MULTIPLY_CHECK_MSG(cond, "", __VA_ARGS__));
#else
#define MULTIPLY_CHECK(cond, ...)                     \
  ::multipyError::detail::multipyCheckFail(                 \
      __func__,                                  \
      __FILE__,                                  \
      static_cast<uint32_t>(__LINE__),           \
      MULTIPLY_CHECK_MSG(cond, "", ##__VA_ARGS__));

#endif // STRIP_ERROR_MESSAGES
#endif // MULTIPY_UTIL_EXCEPTION_H_
