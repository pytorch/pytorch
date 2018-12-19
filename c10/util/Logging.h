#ifndef C10_UTIL_LOGGING_H_
#define C10_UTIL_LOGGING_H_

#include <climits>
#include <exception>
#include <functional>
#include <limits>
#include <sstream>

#include "c10/macros/Macros.h"
#include "c10/util/Exception.h"
#include "c10/util/Flags.h"
#include "c10/util/StringUtil.h"

// CAFFE2_LOG_THRESHOLD is a compile time flag that would allow us to turn off
// logging at compile time so no logging message below that level is produced
// at all. The value should be between INT_MIN and CAFFE_FATAL.
#ifndef CAFFE2_LOG_THRESHOLD
// If we have not defined the compile time log threshold, we keep all the
// log cases.
#define CAFFE2_LOG_THRESHOLD INT_MIN
#endif // CAFFE2_LOG_THRESHOLD

// Below are different implementations for glog and non-glog cases.
#ifdef C10_USE_GLOG
#include "c10/util/logging_is_google_glog.h"
#else // !C10_USE_GLOG
#include "c10/util/logging_is_not_google_glog.h"
#endif // C10_USE_GLOG

C10_DECLARE_int(caffe2_log_level);
C10_DECLARE_bool(caffe2_use_fatal_for_enforce);

// Some versions of GLOG support less-spammy version of LOG_EVERY_MS. If it's
// not available - just short-circuit to the always working one one.
// We define the C10_ name to avoid confusing other files
#ifdef LOG_EVERY_MS
#define C10_LOG_EVERY_MS(severity, ms) LOG_EVERY_MS(severity, ms)
#else
#define C10_LOG_EVERY_MS(severity, ms) LOG(severity)
#endif

namespace c10 {

using std::string;

// Functions that we use for initialization.
C10_API bool InitCaffeLogging(int* argc, char** argv);
C10_API void UpdateLoggingLevelsFromFlags();

C10_API C10_NORETURN void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller = nullptr);

constexpr bool IsUsingGoogleLogging() {
#ifdef C10_USE_GLOG
  return true;
#else
  return false;
#endif
}

/**
 * A utility to allow one to show log info to stderr after the program starts.
 *
 * This is similar to calling GLOG's --logtostderr, or setting caffe2_log_level
 * to smaller than INFO. You are recommended to only use this in a few sparse
 * cases, such as when you want to write a tutorial or something. Normally, use
 * the commandline flags to set the log level.
 */
C10_API void ShowLogInfoToStderr();

C10_API void SetStackTraceFetcher(std::function<string(void)> fetcher);

using EnforceNotMet = ::c10::Error;

#define CAFFE_ENFORCE(condition, ...)                               \
  do {                                                              \
    if (!(condition)) {                                             \
      ::c10::ThrowEnforceNotMet(                                    \
          __FILE__, __LINE__, #condition, ::c10::str(__VA_ARGS__)); \
    }                                                               \
  } while (false)

#define CAFFE_ENFORCE_WITH_CALLER(condition, ...)                         \
  do {                                                                    \
    if (!(condition)) {                                                   \
      ::c10::ThrowEnforceNotMet(                                          \
          __FILE__, __LINE__, #condition, ::c10::str(__VA_ARGS__), this); \
    }                                                                     \
  } while (false)

#define CAFFE_THROW(...) \
  ::c10::ThrowEnforceNotMet(__FILE__, __LINE__, "", ::c10::str(__VA_ARGS__))

/**
 * Rich logging messages
 *
 * CAFFE_ENFORCE_THAT can be used with one of the "checker functions" that
 * capture input argument values and add it to the exception message. E.g.
 * `CAFFE_ENFORCE_THAT(Equals(foo(x), bar(y)), "Optional additional message")`
 * would evaluate both foo and bar only once and if the results are not equal -
 * include them in the exception message.
 *
 * Some of the basic checker functions like Equals or Greater are already
 * defined below. Other header might define customized checkers by adding
 * functions to caffe2::enforce_detail namespace. For example:
 *
 *   namespace caffe2 { namespace enforce_detail {
 *   inline EnforceFailMessage IsVector(const vector<int64_t>& shape) {
 *     if (shape.size() == 1) { return EnforceOK(); }
 *     return c10::str("Shape ", shape, " is not a vector");
 *   }
 *   }}
 *
 * With further usages like `CAFFE_ENFORCE_THAT(IsVector(Input(0).dims()))`
 *
 * Convenient wrappers for binary operations like CAFFE_ENFORCE_EQ are provided
 * too. Please use them instead of CHECK_EQ and friends for failures in
 * user-provided input.
 */

namespace enforce_detail {

struct C10_API EnforceOK {};

class C10_API EnforceFailMessage {
 public:
#ifdef _MSC_VER
  // MSVC + NVCC ignores constexpr and will issue a warning if included.
  /* implicit */ EnforceFailMessage(EnforceOK) : msg_(nullptr) {}
#else
  constexpr /* implicit */ EnforceFailMessage(EnforceOK) : msg_(nullptr) {}
#endif
  EnforceFailMessage(EnforceFailMessage&&) = default;
  EnforceFailMessage(const EnforceFailMessage&) = delete;
  EnforceFailMessage& operator=(EnforceFailMessage&&) = delete;
  EnforceFailMessage& operator=(const EnforceFailMessage&) = delete;

  // Catch all wrong usages like CAFFE_ENFORCE_THAT(x < y)
  template <class... Args>
  /* implicit */ EnforceFailMessage(Args...) {
    static_assert(
        // This stands for an "impossible" condition. Plain `false` doesn't
        // trick compiler enough.
        sizeof...(Args) == std::numeric_limits<std::size_t>::max(),
        "CAFFE_ENFORCE_THAT has to be used with one of special check functions "
        "like `Equals`. Use CAFFE_ENFORCE for simple boolean checks.");
  }

  /* implicit */ EnforceFailMessage(std::string&& msg);

  inline bool bad() const {
    return msg_ != nullptr;
  }
  std::string get_message_and_free(std::string&& extra) const {
    std::string r;
    if (extra.empty()) {
      r = std::move(*msg_);
    } else {
      r = ::c10::str(std::move(*msg_), ". ", std::move(extra));
    }
    delete msg_;
    return r;
  }

 private:
  std::string* msg_;
};

#define BINARY_COMP_HELPER(name, op)                         \
  template <typename T1, typename T2>                        \
  inline EnforceFailMessage name(const T1& x, const T2& y) { \
    if (x op y) {                                            \
      return EnforceOK();                                    \
    }                                                        \
    return c10::str(x, " vs ", y);                           \
  }
BINARY_COMP_HELPER(Equals, ==)
BINARY_COMP_HELPER(NotEquals, !=)
BINARY_COMP_HELPER(Greater, >)
BINARY_COMP_HELPER(GreaterEquals, >=)
BINARY_COMP_HELPER(Less, <)
BINARY_COMP_HELPER(LessEquals, <=)
#undef BINARY_COMP_HELPER

#define CAFFE_ENFORCE_THAT_IMPL(condition, expr, ...)                   \
  do {                                                                  \
    using namespace ::c10::enforce_detail;                              \
    const EnforceFailMessage& CAFFE_ENFORCE_THAT_IMPL_r_ = (condition); \
    if (CAFFE_ENFORCE_THAT_IMPL_r_.bad()) {                             \
      ::c10::ThrowEnforceNotMet(                                        \
          __FILE__,                                                     \
          __LINE__,                                                     \
          expr,                                                         \
          CAFFE_ENFORCE_THAT_IMPL_r_.get_message_and_free(              \
              ::c10::str(__VA_ARGS__)));                                \
    }                                                                   \
  } while (false)

#define CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(condition, expr, ...)      \
  do {                                                                 \
    using namespace ::c10::enforce_detail;                             \
    const EnforceFailMessage& CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER_r_ = \
        (condition);                                                   \
    if (CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER_r_.bad()) {                \
      ::c10::ThrowEnforceNotMet(                                       \
          __FILE__,                                                    \
          __LINE__,                                                    \
          expr,                                                        \
          CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER_r_.get_message_and_free( \
              ::c10::str(__VA_ARGS__)),                                \
          this);                                                       \
    }                                                                  \
  } while (false)
} // namespace enforce_detail

#define CAFFE_ENFORCE_THAT(condition, ...) \
  CAFFE_ENFORCE_THAT_IMPL((condition), #condition, __VA_ARGS__)

#define CAFFE_ENFORCE_EQ(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL(Equals((x), (y)), #x " == " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_NE(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL(NotEquals((x), (y)), #x " != " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_LE(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL(LessEquals((x), (y)), #x " <= " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_LT(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL(Less((x), (y)), #x " < " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_GE(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL(GreaterEquals((x), (y)), #x " >= " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_GT(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL(Greater((x), (y)), #x " > " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_EQ_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(          \
      Equals((x), (y)), #x " == " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_NE_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(          \
      NotEquals((x), (y)), #x " != " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_LE_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(          \
      LessEquals((x), (y)), #x " <= " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_LT_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(Less((x), (y)), #x " < " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_GE_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(          \
      GreaterEquals((x), (y)), #x " >= " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_GT_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(          \
      Greater((x), (y)), #x " > " #y, __VA_ARGS__)
} // namespace c10

#endif // C10_UTIL_LOGGING_H_
