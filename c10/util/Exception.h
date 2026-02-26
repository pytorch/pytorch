#ifndef C10_UTIL_EXCEPTION_H_
#define C10_UTIL_EXCEPTION_H_

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/Backtrace.h>
#include <c10/util/Lazy.h>
#include <c10/util/StringUtil.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__
#endif

namespace c10 {

/// The primary ATen error class.
/// Provides a complete error message with source location information via
/// `what()`, and a more concise message via `what_without_backtrace()`.
/// Don't throw this directly; use TORCH_CHECK/TORCH_INTERNAL_ASSERT instead.
///
/// NB: c10::Error is handled specially by the default torch to suppress the
/// backtrace, see torch/csrc/Exceptions.h
class C10_API Error : public std::exception {
 private:
  // The actual error message.
  std::string msg_;

  // Context for the message (in order of decreasing specificity).  Context will
  // be automatically formatted appropriately, so it is not necessary to add
  // extra leading/trailing newlines to strings inside this vector
  std::vector<std::string> context_;

  // The C++ backtrace at the point when this exception was raised.  This
  // may be empty if there is no valid backtrace.  (We don't use optional
  // here to reduce the dependencies this file has.)
  Backtrace backtrace_;

  // These two are derived fields from msg_stack_ and backtrace_, but we need
  // fields for the strings so that we can return a const char* (as the
  // signature of std::exception requires).  Currently, the invariant
  // is that these fields are ALWAYS populated consistently with respect
  // to msg_stack_ and backtrace_.
  mutable OptimisticLazy<std::string> what_;
  std::string what_without_backtrace_;

  // This is a little debugging trick: you can stash a relevant pointer
  // in caller, and then when you catch the exception, you can compare
  // against pointers you have on hand to get more information about
  // where the exception came from.  In Caffe2, this is used to figure
  // out which operator raised an exception.
  const void* caller_;

 public:
  // PyTorch-style Error constructor.  NB: the implementation of this
  // is actually in Logging.cpp
  Error(SourceLocation source_location, std::string msg);

  // Caffe2-style error message
  Error(
      const char* file,
      const uint32_t line,
      const char* condition,
      const std::string& msg,
      Backtrace backtrace,
      const void* caller = nullptr);

  // Base constructor
  Error(
      std::string msg,
      Backtrace backtrace = nullptr,
      const void* caller = nullptr);

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

  const Backtrace& backtrace() const;

  /// Returns the complete error message, including the source location.
  /// The returned pointer is invalidated if you call add_context() on
  /// this object.
  const char* what() const noexcept override;

  const void* caller() const noexcept {
    return caller_;
  }

  /// Returns only the error message string, without source location.
  /// The returned pointer is invalidated if you call add_context() on
  /// this object.
  virtual const char* what_without_backtrace() const noexcept {
    return what_without_backtrace_.c_str();
  }

 private:
  void refresh_what();
  std::string compute_what(bool include_backtrace) const;
};

class C10_API Warning {
 public:
  class C10_API UserWarning{};
  class C10_API DeprecationWarning{};

  using warning_variant_t = std::variant<UserWarning, DeprecationWarning>;

  Warning(
      warning_variant_t type,
      const SourceLocation& source_location,
      std::string msg,
      bool verbatim);

  Warning(
      warning_variant_t type,
      SourceLocation source_location,
      const char* msg,
      bool verbatim);

  Warning(
      warning_variant_t type,
      SourceLocation source_location,
      ::c10::detail::CompileTimeEmptyString msg,
      bool verbatim);

  // Getters for members
  warning_variant_t type() const;
  const SourceLocation& source_location() const;
  const std::string& msg() const;
  bool verbatim() const;

 private:
  // The type of warning
  warning_variant_t type_;

  // Where the warning happened.
  SourceLocation source_location_;

  // The actual warning message.
  std::string msg_;

  // See note: [Verbatim Warnings]
  bool verbatim_;
};

using UserWarning = Warning::UserWarning;
using DeprecationWarning = Warning::DeprecationWarning;

// Issue a warning with a given message. Dispatched to the current
// warning handler.
void C10_API warn(const Warning& warning);

class C10_API WarningHandler {
 public:
  virtual ~WarningHandler() = default;
  /// The default warning handler. Prints the message to stderr.
  virtual void process(const Warning& warning);
};

namespace WarningUtils {

// Note: [Verbatim Warnings]
// Warnings originating in C++ code can appear out-of-place to Python users:
// a user runs a line in Python, but the warning references a line in C++.
// Some parts of PyTorch, like the JIT, are cognizant of this mismatch
// and take care to map warnings back to the user's program, but most
// of PyTorch simply throws a context-free warning. To allow warning
// handlers to add context where appropriate, warn takes the
// "verbatim" flag. When this is false a warning handler might append
// the C++ warning to a Python warning message that relates the warning
// back to the user's program. Callers who have already accounted for
// context in their warnings should set verbatim to true so their warnings
// appear without modification.

/// Sets the global warning handler. This is not thread-safe, so it should
/// generally be called once during initialization or while holding the GIL
/// for programs that use python.
/// User is responsible for keeping the WarningHandler alive until
/// it is not needed.
C10_API void set_warning_handler(WarningHandler* handler) noexcept(true);
/// Gets the global warning handler.
C10_API WarningHandler* get_warning_handler() noexcept(true);

class C10_API WarningHandlerGuard {
  WarningHandler* prev_handler_;

 public:
  WarningHandlerGuard(WarningHandler* new_handler)
      : prev_handler_(c10::WarningUtils::get_warning_handler()) {
    c10::WarningUtils::set_warning_handler(new_handler);
  }
  WarningHandlerGuard(WarningHandlerGuard&& other) = delete;
  WarningHandlerGuard(const WarningHandlerGuard&) = delete;
  WarningHandlerGuard& operator=(const WarningHandlerGuard&) = delete;
  WarningHandlerGuard& operator=(WarningHandlerGuard&&) = delete;
  ~WarningHandlerGuard() {
    c10::WarningUtils::set_warning_handler(prev_handler_);
  }
};

/// The TORCH_WARN_ONCE macro is difficult to test for. Use
/// setWarnAlways(true) to turn it into TORCH_WARN, which can be
/// tested for more easily.
C10_API void set_warnAlways(bool /*setting*/) noexcept(true);
C10_API bool get_warnAlways() noexcept(true);

// A RAII guard that sets warn_always (not thread-local) on
// construction, and sets it back to the original value upon destruction.
struct C10_API WarnAlways {
 public:
  explicit WarnAlways(bool setting = true);
  ~WarnAlways();

 private:
  bool prev_setting;
};

} // namespace WarningUtils

// Like Error, but we always report the C++ backtrace, instead of only
// reporting when TORCH_SHOW_CPP_STACKTRACES
class C10_API ErrorAlwaysShowCppStacktrace : public Error {
  using Error::Error;
  const char* what_without_backtrace() const noexcept override {
    return what();
  }
};

// Used in ATen for out-of-bound indices that can reasonably only be detected
// lazily inside a kernel (See: advanced indexing).  These turn into
// IndexError when they cross to Python.
class C10_API IndexError : public Error {
  using Error::Error;
};

// Used in ATen for invalid values.  These turn into
// ValueError when they cross to Python.
class C10_API ValueError : public Error {
  using Error::Error;
};

// Used in ATen for invalid types.  These turn into
// TypeError when they cross to Python.
class C10_API TypeError : public Error {
  using Error::Error;
};

// Used in ATen for functionality that is not implemented.  These turn into
// NotImplementedError when they cross to Python.
class C10_API NotImplementedError : public Error {
  using Error::Error;
};

// Used in ATen for buffer-related errors, e.g. trying to create a DLPack of
// an unsupported device.  These turn into BufferError when they cross to
// Python.
class C10_API BufferError : public Error {
  using Error::Error;
};

// Used in ATen for non finite indices.  These turn into
// ExitException when they cross to Python.
class C10_API EnforceFiniteError : public Error {
  using Error::Error;
};

// Used in Onnxifi backend lowering.  These turn into
// ExitException when they cross to Python.
class C10_API OnnxfiBackendSystemError : public Error {
  using Error::Error;
};

// Used for numerical errors from the linalg module. These
// turn into LinAlgError when they cross into Python.
class C10_API LinAlgError : public Error {
  using Error::Error;
};

class C10_API OutOfMemoryError : public Error {
  using Error::Error;
};

// Used for handling syntactic errors in input arguments.
// These turn into SyntaxError when the cross into Python.
class C10_API SyntaxError : public Error {
  using Error::Error;
};

// Raised when accelerator API call hits an error.
// These turn into AcceleratorError when the cross into Python
class C10_API AcceleratorError : public Error {
  int32_t error_code;

 public:
  AcceleratorError(SourceLocation loc, int32_t code, const std::string& msg)
      : Error(loc, msg), error_code(code) {}
  int32_t get_error_code() const {
    return error_code;
  }
};

// Base error type for all distributed errors.
// These turn into DistError when they cross into Python.
class C10_API DistError : public Error {
  using Error::Error;
};

// Used for collective communication library errors from the distributed module.
// These turn into DistBackendError when they cross into Python.
class C10_API DistBackendError : public DistError {
  using DistError::DistError;
};

// Used for errors originating from the store.
// These turn into DistStoreError when they cross into Python.
class C10_API DistStoreError : public DistError {
  using DistError::DistError;
};

// Used for errors originating from the TCP/IP stack and not from collective
// libraries. These turn into DistNetworkError when they cross into Python.
class C10_API DistNetworkError : public DistError {
  using DistError::DistError;
};

// Raised when a queue is empty and a non-blocking pop is called.
// Translated to torch.distributed.QueueEmptyError in Python
class C10_API DistQueueEmptyError : public DistStoreError {
  using DistStoreError::DistStoreError;
};

// A utility function to return an exception std::string by prepending its
// exception type before its what() content
C10_API std::string GetExceptionString(const std::exception& e);

} // namespace c10

// Private helper macro for implementing TORCH_INTERNAL_ASSERT and TORCH_CHECK
//
// Note: In the debug build With MSVC, __LINE__ might be of long type (a.k.a
// int32_t), which is different from the definition of `SourceLocation` that
// requires unsigned int (a.k.a uint32_t) and may cause a compile error with the
// message: error C2397: conversion from 'long' to 'uint32_t' requires a
// narrowing conversion Here the static cast is used to pass the build. if this
// is used inside a lambda the __func__ macro expands to operator(), which isn't
// very useful, but hard to fix in a macro so suppressing the warning.
#define C10_THROW_ERROR(err_type, msg) \
  throw ::c10::err_type(               \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, msg)

#define C10_BUILD_ERROR(err_type, msg) \
  ::c10::err_type({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, msg)

// Private helper macro for workaround MSVC misexpansion of nested macro
// invocations involving __VA_ARGS__.  See
// https://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
#define C10_EXPAND_MSVC_WORKAROUND(x) x

#include <torch/headeronly/util/Exception.h>

// ----------------------------------------------------------------------------
// Error reporting macros
// ----------------------------------------------------------------------------

#ifdef STRIP_ERROR_MESSAGES
#define TORCH_RETHROW(e, ...)                       \
  do {                                              \
    (void)e; /* Suppress unused variable warning */ \
    throw;                                          \
  } while (false)
#else
#define TORCH_RETHROW(e, ...)               \
  do {                                      \
    e.add_context(::c10::str(__VA_ARGS__)); \
    throw;                                  \
  } while (false)
#endif

// A utility macro to provide assert()-like functionality; that is, enforcement
// of internal invariants in code.  It supports an arbitrary number of extra
// arguments (evaluated only on failure), which will be printed in the assert
// failure message using operator<< (this is useful to print some variables
// which may be useful for debugging.)
//
// Usage:
//    TORCH_INTERNAL_ASSERT(should_be_true);
//    TORCH_INTERNAL_ASSERT(x == 0, "x = ", x);
//
// Assuming no bugs in PyTorch, the conditions tested by this macro should
// always be true; e.g., it should be possible to disable all of these
// conditions without changing observable user behavior.  If you would like to
// do error reporting for user input, please use TORCH_CHECK instead.
//
// NOTE: It is SAFE to use this macro in production code; on failure, this
// simply raises an exception, it does NOT unceremoniously quit the process
// (unlike assert()).
//
#ifdef STRIP_ERROR_MESSAGES
#define TORCH_INTERNAL_ASSERT(cond, ...)                              \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                               \
    ::c10::detail::torchCheckFail(                                    \
        __func__,                                                     \
        __FILE__,                                                     \
        static_cast<uint32_t>(__LINE__),                              \
        #cond " INTERNAL ASSERT FAILED at " C10_STRINGIZE(__FILE__)); \
  }
#else
// It would be nice if we could build a combined string literal out of
// the TORCH_INTERNAL_ASSERT prefix and a user-provided string literal
// as the first argument, but there doesn't seem to be any good way to
// do that while still supporting having a first argument that isn't a
// string literal.
#define TORCH_INTERNAL_ASSERT(cond, ...)                                         \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                                          \
    ::c10::detail::torchInternalAssertFail(                                      \
        __func__,                                                                \
        __FILE__,                                                                \
        static_cast<uint32_t>(__LINE__),                                         \
        #cond                                                                    \
        " INTERNAL ASSERT FAILED at " C10_STRINGIZE(__FILE__) ":" C10_STRINGIZE( \
            __LINE__) ", please report a bug to PyTorch. ",                      \
        c10::str(__VA_ARGS__));                                                  \
  }
#endif

// A utility macro to make it easier to test for error conditions from user
// input.  Like TORCH_INTERNAL_ASSERT, it supports an arbitrary number of extra
// arguments (evaluated only on failure), which will be printed in the error
// message using operator<< (e.g., you can pass any object which has
// operator<< defined.  Most objects in PyTorch have these definitions!)
//
// Usage:
//    TORCH_CHECK(should_be_true); // A default error message will be provided
//                                 // in this case; but we recommend writing an
//                                 // explicit error message, as it is more
//                                 // user friendly.
//    TORCH_CHECK(x == 0, "Expected x to be 0, but got ", x);
//
// On failure, this macro will raise an exception.  If this exception propagates
// to Python, it will convert into a Python RuntimeError.
//
// NOTE: It is SAFE to use this macro in production code; on failure, this
// simply raises an exception, it does NOT unceremoniously quit the process
// (unlike CHECK() from glog.)
//
#define TORCH_CHECK_WITH(error_t, cond, ...) \
  TORCH_CHECK_WITH_MSG(error_t, cond, "", __VA_ARGS__)

#ifdef STRIP_ERROR_MESSAGES
#define TORCH_CHECK_MSG(cond, type, ...) \
  (#cond #type " CHECK FAILED at " C10_STRINGIZE(__FILE__))
#define TORCH_CHECK_WITH_MSG(error_t, cond, type, ...)                \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                               \
    C10_THROW_ERROR(Error, TORCH_CHECK_MSG(cond, type, __VA_ARGS__)); \
  }
#else

namespace c10::detail {
template <typename... Args>
auto torchCheckMsgImpl(const char* /*msg*/, const Args&... args) {
  return ::c10::str(args...);
}
inline C10_API const char* torchCheckMsgImpl(const char* msg) {
  return msg;
}
// If there is just 1 user-provided C-string argument, use it.
inline C10_API const char* torchCheckMsgImpl(
    const char* /*msg*/,
    const char* args) {
  return args;
}
} // namespace c10::detail

#define TORCH_CHECK_MSG(cond, type, ...)                   \
  (::c10::detail::torchCheckMsgImpl(                       \
      "Expected " #cond                                    \
      " to be true, but got false.  "                      \
      "(Could this error message be improved?  If so, "    \
      "please report an enhancement request to PyTorch.)", \
      ##__VA_ARGS__))
#define TORCH_CHECK_WITH_MSG(error_t, cond, type, ...)                  \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                                 \
    C10_THROW_ERROR(error_t, TORCH_CHECK_MSG(cond, type, __VA_ARGS__)); \
  }
#endif

namespace c10::detail {

[[noreturn]] C10_API void torchCheckFail(
    const char* func,
    const char* file,
    uint32_t line,
    const std::string& msg);
[[noreturn]] C10_API void torchCheckFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg);

// The c10::str() call that creates userMsg can have 1 of 3 return
// types depending on the number and types of arguments passed to
// TORCH_INTERNAL_ASSERT.  0 arguments will get a
// CompileTimeEmptyString, 1 const char * will be passed straight
// through, and anything else will get converted to std::string.
[[noreturn]] C10_API void torchInternalAssertFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    const char* userMsg);
[[noreturn]] inline C10_API void torchInternalAssertFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    ::c10::detail::CompileTimeEmptyString /*userMsg*/) {
  torchCheckFail(func, file, line, condMsg);
}
[[noreturn]] C10_API void torchInternalAssertFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    const std::string& userMsg);

} // namespace c10::detail

#ifdef STANDALONE_TORCH_HEADER

// TORCH_CHECK throws std::runtime_error instead of c10::Error which is
// useful when certain headers are used in a libtorch-independent way,
// e.g. when Vectorized<T> is used in AOTInductor generated code.
#ifdef STRIP_ERROR_MESSAGES
#define TORCH_CHECK(cond, ...)                \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {       \
    throw std::runtime_error(TORCH_CHECK_MSG( \
        cond,                                 \
        "",                                   \
        __func__,                             \
        ", ",                                 \
        __FILE__,                             \
        ":",                                  \
        __LINE__,                             \
        ", ",                                 \
        __VA_ARGS__));                        \
  }
#else
#define TORCH_CHECK(cond, ...)                \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {       \
    throw std::runtime_error(TORCH_CHECK_MSG( \
        cond,                                 \
        "",                                   \
        __func__,                             \
        ", ",                                 \
        __FILE__,                             \
        ":",                                  \
        __LINE__,                             \
        ", ",                                 \
        ##__VA_ARGS__));                      \
  }
#endif

#else

#ifdef STRIP_ERROR_MESSAGES
#define TORCH_CHECK(cond, ...)                   \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {          \
    ::c10::detail::torchCheckFail(               \
        __func__,                                \
        __FILE__,                                \
        static_cast<uint32_t>(__LINE__),         \
        TORCH_CHECK_MSG(cond, "", __VA_ARGS__)); \
  }
#else
#define TORCH_CHECK(cond, ...)                     \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {            \
    ::c10::detail::torchCheckFail(                 \
        __func__,                                  \
        __FILE__,                                  \
        static_cast<uint32_t>(__LINE__),           \
        TORCH_CHECK_MSG(cond, "", ##__VA_ARGS__)); \
  }
#endif

#endif

// An utility macro that does what `TORCH_CHECK` does if compiled in the host
// code, otherwise does nothing. Supposed to be used in the code shared between
// host and device code as an alternative for `TORCH_CHECK`.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define TORCH_CHECK_IF_NOT_ON_CUDA(cond, ...)
#else
#define TORCH_CHECK_IF_NOT_ON_CUDA(cond, ...) TORCH_CHECK(cond, ##__VA_ARGS__)
#endif

// Debug only version of TORCH_INTERNAL_ASSERT. This macro only checks in debug
// build, and does nothing in release build.  It is appropriate to use
// in situations where you want to add an assert to a hotpath, but it is
// too expensive to run this assert on production builds.
#ifdef NDEBUG
// Optimized version - generates no code.
#define TORCH_INTERNAL_ASSERT_DEBUG_ONLY(...) \
  while (false)                               \
  C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(__VA_ARGS__))
#else
#define TORCH_INTERNAL_ASSERT_DEBUG_ONLY(...) \
  C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(__VA_ARGS__))
#endif

// TODO: We're going to get a lot of similar looking string literals
// this way; check if this actually affects binary size.

// Like TORCH_CHECK, but raises LinAlgError instead of Error.
#define TORCH_CHECK_LINALG(cond, ...) \
  TORCH_CHECK_WITH_MSG(LinAlgError, cond, "LINALG", __VA_ARGS__)

// Like TORCH_CHECK, but raises IndexErrors instead of Errors.
#define TORCH_CHECK_INDEX(cond, ...) \
  TORCH_CHECK_WITH_MSG(IndexError, cond, "INDEX", __VA_ARGS__)

// Like TORCH_CHECK, but raises ValueErrors instead of Errors.
#define TORCH_CHECK_VALUE(cond, ...) \
  TORCH_CHECK_WITH_MSG(ValueError, cond, "VALUE", __VA_ARGS__)

// Like TORCH_CHECK, but raises TypeErrors instead of Errors.
#define TORCH_CHECK_TYPE(cond, ...) \
  TORCH_CHECK_WITH_MSG(TypeError, cond, "TYPE", __VA_ARGS__)

// Like TORCH_CHECK, but raises NotImplementedErrors instead of Errors.
#define TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) \
  TORCH_CHECK_WITH_MSG(NotImplementedError, cond, "TYPE", __VA_ARGS__)

// Like TORCH_CHECK, but raises BufferError instead of Errors.
#define TORCH_CHECK_BUFFER(cond, ...) \
  TORCH_CHECK_WITH_MSG(BufferError, cond, "TYPE", __VA_ARGS__)

#define TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE(cond, ...) \
  TORCH_CHECK_WITH_MSG(                                   \
      ErrorAlwaysShowCppStacktrace, cond, "TYPE", ##__VA_ARGS__)

#ifdef STRIP_ERROR_MESSAGES
#define WARNING_MESSAGE_STRING(...) \
  ::c10::detail::CompileTimeEmptyString {}
#else
#define WARNING_MESSAGE_STRING(...) ::c10::str(__VA_ARGS__)
#endif

// Report a warning to the user.  Accepts an arbitrary number of extra
// arguments which are concatenated into the warning message using operator<<
//
#ifdef DISABLE_WARN
#define _TORCH_WARN_WITH(...) ((void)0);
#else
#define _TORCH_WARN_WITH(warning_t, ...)                     \
  ::c10::warn(::c10::Warning(                                \
      warning_t(),                                           \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
      WARNING_MESSAGE_STRING(__VA_ARGS__),                   \
      false));
#endif

#define TORCH_WARN(...) _TORCH_WARN_WITH(::c10::UserWarning, __VA_ARGS__);

#define TORCH_WARN_DEPRECATION(...) \
  _TORCH_WARN_WITH(::c10::DeprecationWarning, __VA_ARGS__);

// Report a warning to the user only once.  Accepts an arbitrary number of extra
// arguments which are concatenated into the warning message using operator<<
//
#define _TORCH_WARN_ONCE(...)                                \
  [[maybe_unused]] static const auto C10_ANONYMOUS_VARIABLE( \
      torch_warn_once_) = [&] {                              \
    TORCH_WARN(__VA_ARGS__);                                 \
    return true;                                             \
  }()

#ifdef DISABLE_WARN
#define TORCH_WARN_ONCE(...) ((void)0);
#else
#define TORCH_WARN_ONCE(...)                   \
  if (::c10::WarningUtils::get_warnAlways()) { \
    TORCH_WARN(__VA_ARGS__);                   \
  } else {                                     \
    _TORCH_WARN_ONCE(__VA_ARGS__);             \
  }
#endif

// Report an error with a specific argument
// NOTE: using the argument name in TORCH_CHECK's message is preferred
#define TORCH_CHECK_ARG(cond, argN, ...) \
  TORCH_CHECK(cond, "invalid argument ", argN, ": ", __VA_ARGS__)

#ifndef FATAL_IF
#ifdef C10_USE_GLOG
#define FATAL_IF(condition)                                           \
  condition ? (void)0                                                 \
            : ::c10::LoggerVoidify() &                                \
          ::c10::MessageLogger(                                       \
              ::c10::SourceLocation::current(), ::google::GLOG_FATAL) \
              .stream()
#else
#define FATAL_IF(condition)                                        \
  condition ? (void)0                                              \
            : ::c10::LoggerVoidify() &                             \
          ::c10::MessageLogger(                                    \
              ::c10::SourceLocation::current(), ::c10::GLOG_FATAL) \
              .stream()
#endif
#endif

#ifndef NON_FATAL_IF
#ifdef C10_USE_GLOG
#define NON_FATAL_IF(condition)                                              \
  condition ? (void)0                                                        \
            : ::c10::LoggerVoidify() &                                       \
          ::c10::MessageLogger(                                              \
              ::c10::SourceLocation::current(), ::google::GLOG_FATAL, false) \
              .stream()
#else
#define NON_FATAL_IF(condition)                                           \
  condition ? (void)0                                                     \
            : ::c10::LoggerVoidify() &                                    \
          ::c10::MessageLogger(                                           \
              ::c10::SourceLocation::current(), ::c10::GLOG_FATAL, false) \
              .stream()
#endif
#endif

// Binary comparison check macros
#define TORCH_CHECK_OP(val1, val2, op)                                      \
  NON_FATAL_IF(((val1)op(val2)))                                            \
      << "Check failed: " #val1 " " #op " " #val2 " (" << (val1) << " vs. " \
      << (val2) << "). "

#define TORCH_DCHECK_OP(val1, val2, op)                                       \
  FATAL_IF(((val1)op(val2))) << "Check failed: " #val1 " " #op " " #val2 " (" \
                             << (val1) << " vs. " << (val2) << "). "

#define TORCH_CHECK_EQ(val1, val2) TORCH_CHECK_OP(val1, val2, ==)
#define TORCH_CHECK_NE(val1, val2) TORCH_CHECK_OP(val1, val2, !=)
#define TORCH_CHECK_LE(val1, val2) TORCH_CHECK_OP(val1, val2, <=)
#define TORCH_CHECK_LT(val1, val2) TORCH_CHECK_OP(val1, val2, <)
#define TORCH_CHECK_GE(val1, val2) TORCH_CHECK_OP(val1, val2, >=)
#define TORCH_CHECK_GT(val1, val2) TORCH_CHECK_OP(val1, val2, >)

// Debug versions of TORCH_CHECK_OP macros
#ifndef NDEBUG
#define TORCH_DCHECK_EQ(val1, val2) TORCH_DCHECK_OP(val1, val2, ==)
#define TORCH_DCHECK_NE(val1, val2) TORCH_DCHECK_OP(val1, val2, !=)
#define TORCH_DCHECK_LE(val1, val2) TORCH_DCHECK_OP(val1, val2, <=)
#define TORCH_DCHECK_LT(val1, val2) TORCH_DCHECK_OP(val1, val2, <)
#define TORCH_DCHECK_GE(val1, val2) TORCH_DCHECK_OP(val1, val2, >=)
#define TORCH_DCHECK_GT(val1, val2) TORCH_DCHECK_OP(val1, val2, >)
#else // !NDEBUG
// Optimized versions - generate no code
#define TORCH_DCHECK_EQ(val1, val2) \
  while (false)                     \
  TORCH_DCHECK_OP(val1, val2, ==)
#define TORCH_DCHECK_NE(val1, val2) \
  while (false)                     \
  TORCH_DCHECK_OP(val1, val2, !=)
#define TORCH_DCHECK_LE(val1, val2) \
  while (false)                     \
  TORCH_DCHECK_OP(val1, val2, <=)
#define TORCH_DCHECK_LT(val1, val2) \
  while (false)                     \
  TORCH_DCHECK_OP(val1, val2, <)
#define TORCH_DCHECK_GE(val1, val2) \
  while (false)                     \
  TORCH_DCHECK_OP(val1, val2, >=)
#define TORCH_DCHECK_GT(val1, val2) \
  while (false)                     \
  TORCH_DCHECK_OP(val1, val2, >)
#endif // NDEBUG

// Null pointer check macro
#define TORCH_CHECK_NOTNULL(val) \
  ::c10::CheckNotNull(__FILE__, __LINE__, #val, (val), false)

#ifndef NDEBUG
#define TORCH_DCHECK_NOTNULL(val) \
  ::c10::CheckNotNull(__FILE__, __LINE__, #val, (val), true)
#else // !NDEBUG
#define TORCH_DCHECK_NOTNULL(val) \
  while (false)                   \
  TORCH_CHECK_NOTNULL(val)
#endif // NDEBUG

// ----------------------------------------------------------------------------
// Deprecated macros
// ----------------------------------------------------------------------------

namespace c10::detail {

/*
// Deprecation disabled until we fix sites in our codebase
[[deprecated("AT_ERROR(msg) is deprecated, use TORCH_CHECK(false, msg)
instead.")]]
*/
inline void deprecated_AT_ERROR() {}

/*
// Deprecation disabled until we fix sites in our codebase
[[deprecated("AT_ASSERT is deprecated, if you mean to indicate an
internal invariant failure, use " \
                       "TORCH_INTERNAL_ASSERT instead; if you mean to do user
error checking, use " \ "TORCH_CHECK.  See
https://github.com/pytorch/pytorch/issues/20287 for more details.")]]
*/
inline void deprecated_AT_ASSERT() {}

/*
// Deprecation disabled until we fix sites in our codebase
[[deprecated("AT_ASSERTM is deprecated, if you mean to indicate an
internal invariant failure, use " \
                       "TORCH_INTERNAL_ASSERT instead; if you mean to do user
error checking, use " \ "TORCH_CHECK.  See
https://github.com/pytorch/pytorch/issues/20287 for more details.")]]
*/
inline void deprecated_AT_ASSERTM() {}

} // namespace c10::detail

// Deprecated alias; this alias was deprecated because people kept mistakenly
// using it for user error checking.  Use TORCH_INTERNAL_ASSERT or TORCH_CHECK
// instead. See https://github.com/pytorch/pytorch/issues/20287 for more
// details.
#define AT_ASSERT(...)                                              \
  do {                                                              \
    ::c10::detail::deprecated_AT_ASSERT();                          \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(__VA_ARGS__)); \
  } while (false)

// Deprecated alias, like AT_ASSERT.  The new TORCH_INTERNAL_ASSERT macro
// supports both 0-ary and variadic calls, so having a separate
// message-accepting macro is not necessary.
//
// NB: we MUST include cond explicitly here, as MSVC will miscompile the macro
// expansion, shunting all of __VA_ARGS__ to cond.  An alternate workaround
// can be seen at
// https://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
#define AT_ASSERTM(cond, ...)                                             \
  do {                                                                    \
    ::c10::detail::deprecated_AT_ASSERTM();                               \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_INTERNAL_ASSERT(cond, __VA_ARGS__)); \
  } while (false)

// Deprecated alias; this alias was deprecated because it represents extra API
// surface that makes it hard for people to understand what macro to use.
// Use TORCH_CHECK(false, ...) or TORCH_INTERNAL_ASSERT(false, ...) to
// unconditionally fail at a line of code.
#define AT_ERROR(...)                                                        \
  do {                                                                       \
    ::c10::detail::deprecated_AT_ERROR();                                    \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_CHECK(false, ::c10::str(__VA_ARGS__))); \
  } while (false)

#endif // C10_UTIL_EXCEPTION_H_
