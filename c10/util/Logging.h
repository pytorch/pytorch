#ifndef C10_UTIL_LOGGING_H_
#define C10_UTIL_LOGGING_H_

#include <climits>
#include <exception>
#include <functional>
#include <limits>
#include <sstream>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Flags.h>
#include <c10/util/StringUtil.h>

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
#include <c10/util/logging_is_google_glog.h>
#else // !C10_USE_GLOG
#include <c10/util/logging_is_not_google_glog.h>
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

// Same for LOG_FIRST_N
#ifdef LOG_FIRST_N
#define C10_LOG_FIRST_N(severity, n) LOG_FIRST_N(severity, n)
#else
#define C10_LOG_FIRST_N(severity, n) LOG(severity)
#endif

// Same for LOG_EVERY_N
#ifdef LOG_EVERY_N
#define C10_LOG_EVERY_N(severity, n) LOG_EVERY_N(severity, n)
#else
#define C10_LOG_EVERY_N(severity, n) LOG(severity)
#endif

namespace c10 {

using std::string;

// Functions that we use for initialization.
C10_API bool InitCaffeLogging(int* argc, char** argv);
C10_API void UpdateLoggingLevelsFromFlags();

[[noreturn]] C10_API void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller = nullptr);

[[noreturn]] C10_API void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const char* msg,
    const void* caller = nullptr);

[[noreturn]] C10_API inline void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    detail::CompileTimeEmptyString /*msg*/,
    const void* caller = nullptr) {
  ThrowEnforceNotMet(file, line, condition, "", caller);
}

[[noreturn]] C10_API void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller = nullptr);

[[noreturn]] C10_API void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    const char* condition,
    const char* msg,
    const void* caller = nullptr);

[[noreturn]] C10_API inline void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    const char* condition,
    detail::CompileTimeEmptyString /*msg*/,
    const void* caller = nullptr) {
  ThrowEnforceFiniteNotMet(file, line, condition, "", caller);
}

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
    if (C10_UNLIKELY(!(condition))) {                               \
      ::c10::ThrowEnforceNotMet(                                    \
          __FILE__, __LINE__, #condition, ::c10::str(__VA_ARGS__)); \
    }                                                               \
  } while (false)

#define CAFFE_ENFORCE_FINITE(condition, ...)                        \
  do {                                                              \
    if (C10_UNLIKELY(!(condition))) {                               \
      ::c10::ThrowEnforceFiniteNotMet(                              \
          __FILE__, __LINE__, #condition, ::c10::str(__VA_ARGS__)); \
    }                                                               \
  } while (false)

#define CAFFE_ENFORCE_WITH_CALLER(condition, ...)                         \
  do {                                                                    \
    if (C10_UNLIKELY(!(condition))) {                                     \
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
 * too. Please use them instead of TORCH_CHECK_EQ and friends for failures in
 * user-provided input.
 */

namespace enforce_detail {

template <typename T1, typename T2>
std::string enforceFailMsgImpl(const T1& x, const T2& y) {
  return c10::str(x, " vs ", y);
}

template <typename T1, typename T2, typename... Args>
std::string enforceFailMsgImpl(const T1& x, const T2& y, const Args&... args) {
  return c10::str(x, " vs ", y, ". ", args...);
}

// GCC7 is getting an internal compiler error on the new
// implementation, so keep the old one (which evaluates the error
// message eagerly and therefore is undesirable for general use
// compared to the new one) around for it.
#if defined(__GNUG__) && __GNUC__ <= 7 && !defined(__clang__)
template <typename Pred, typename T1, typename T2, typename... Args>
void enforceThatImpl(
    Pred p,
    const T1& lhs,
    const T2& rhs,
    const char* file,
    int line,
    const char* expr,
    const void* caller,
    const Args&... args) {
  if (C10_UNLIKELY(!(p(lhs, rhs)))) {
    ::c10::ThrowEnforceNotMet(
        file,
        line,
        expr,
        ::c10::enforce_detail::enforceFailMsgImpl(lhs, rhs, args...),
        caller);
  }
}

#define CAFFE_ENFORCE_THAT_IMPL(op, lhs, rhs, expr, ...) \
  ::c10::enforce_detail::enforceThatImpl(                \
      op, lhs, rhs, __FILE__, __LINE__, expr, nullptr, ##__VA_ARGS__)

#define CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(op, lhs, rhs, expr, ...) \
  ::c10::enforce_detail::enforceThatImpl(                            \
      op, (lhs), (rhs), __FILE__, __LINE__, expr, this, ##__VA_ARGS__)

#else
template <typename Pred, typename T1, typename T2, typename GetFailMsgFunc>
void enforceThatImpl(
    Pred p,
    const T1& lhs,
    const T2& rhs,
    const char* file,
    int line,
    const char* expr,
    const void* caller,
    GetFailMsgFunc getFailMsg) {
  if (C10_UNLIKELY(!(p(lhs, rhs)))) {
    ::c10::ThrowEnforceNotMet(file, line, expr, getFailMsg(lhs, rhs), caller);
  }
}

#define CAFFE_ENFORCE_THAT_IMPL(op, lhs, rhs, expr, ...)  \
  ::c10::enforce_detail::enforceThatImpl(                 \
      op,                                                 \
      (lhs),                                              \
      (rhs),                                              \
      __FILE__,                                           \
      __LINE__,                                           \
      expr,                                               \
      nullptr,                                            \
      [&](const auto& arg1, const auto& arg2) {           \
        return ::c10::enforce_detail::enforceFailMsgImpl( \
            arg1, arg2, ##__VA_ARGS__);                   \
      })

#define CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(op, lhs, rhs, expr, ...) \
  ::c10::enforce_detail::enforceThatImpl(                            \
      op,                                                            \
      (lhs),                                                         \
      (rhs),                                                         \
      __FILE__,                                                      \
      __LINE__,                                                      \
      expr,                                                          \
      this,                                                          \
      [&](const auto& arg1, const auto& arg2) {                      \
        return ::c10::enforce_detail::enforceFailMsgImpl(            \
            arg1, arg2, ##__VA_ARGS__);                              \
      })
#endif

} // namespace enforce_detail

#define CAFFE_ENFORCE_THAT(cmp, op, lhs, rhs, ...) \
  CAFFE_ENFORCE_THAT_IMPL(cmp, lhs, rhs, #lhs " " #op " " #rhs, ##__VA_ARGS__)

#define CAFFE_ENFORCE_BINARY_OP(cmp, op, x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL(cmp, x, y, #x " " #op " " #y, ##__VA_ARGS__)
#define CAFFE_ENFORCE_EQ(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP(std::equal_to<void>(), ==, x, y, ##__VA_ARGS__)
#define CAFFE_ENFORCE_NE(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP(std::not_equal_to<void>(), !=, x, y, ##__VA_ARGS__)
#define CAFFE_ENFORCE_LE(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP(std::less_equal<void>(), <=, x, y, ##__VA_ARGS__)
#define CAFFE_ENFORCE_LT(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP(std::less<void>(), <, x, y, ##__VA_ARGS__)
#define CAFFE_ENFORCE_GE(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP(std::greater_equal<void>(), >=, x, y, ##__VA_ARGS__)
#define CAFFE_ENFORCE_GT(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP(std::greater<void>(), >, x, y, ##__VA_ARGS__)

#define CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(cmp, op, x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(                          \
      cmp, x, y, #x " " #op " " #y, ##__VA_ARGS__)
#define CAFFE_ENFORCE_EQ_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          \
      std::equal_to<void>(), ==, x, y, ##__VA_ARGS__)
#define CAFFE_ENFORCE_NE_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          \
      std::not_equal_to<void>(), !=, x, y, ##__VA_ARGS__)
#define CAFFE_ENFORCE_LE_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          \
      std::less_equal<void>(), <=, x, y, ##__VA_ARGS__)
#define CAFFE_ENFORCE_LT_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(std::less<void>(), <, x, y, ##__VA_ARGS__)
#define CAFFE_ENFORCE_GE_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          \
      std::greater_equal<void>(), >=, x, y, ##__VA_ARGS__)
#define CAFFE_ENFORCE_GT_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_BINARY_OP_WITH_CALLER(          \
      std::greater<void>(), >, x, y, ##__VA_ARGS__)

/**
 * Very lightweight logging for the first time API usage. It's beneficial for
 * tracking of individual functionality usage in larger applications.
 *
 * In order to ensure light-weightedness of logging, we utilize static variable
 * trick - LogAPIUsage will be invoked only once and further invocations will
 * just do an atomic check.
 *
 * Example:
 *   // Logs caller info with an arbitrary text event, if there is a usage.
 *   C10_LOG_API_USAGE_ONCE("my_api");
 */
#define C10_LOG_API_USAGE_ONCE(...)                        \
  C10_UNUSED static bool C10_ANONYMOUS_VARIABLE(logFlag) = \
      ::c10::detail::LogAPIUsageFakeReturn(__VA_ARGS__);

// API usage logging capabilities
C10_API void SetAPIUsageLogger(std::function<void(const std::string&)> logger);
C10_API void LogAPIUsage(const std::string& context);

C10_API void SetAPIUsageMetadataLogger(
    std::function<void(
        const std::string&,
        const std::map<std::string, std::string>& metadata_map)> logger);
C10_API void LogAPIUsageMetadata(
    const std::string& context,
    const std::map<std::string, std::string>& metadata_map);

// PyTorch ddp usage logging capabilities
// DDPLoggingData holds data that can be logged in applications
// for analysis and debugging. Data structure is defined in
// c10 directory so that it can be easily imported by both c10
// and torch files.
struct DDPLoggingData {
  // logging fields that are string types.
  std::map<std::string, std::string> strs_map;
  // logging fields that are int64_t types.
  std::map<std::string, int64_t> ints_map;
};

C10_API void SetPyTorchDDPUsageLogger(
    std::function<void(const DDPLoggingData&)> logger);
C10_API void LogPyTorchDDPUsage(const DDPLoggingData& ddpData);

namespace detail {
// Return value is needed to do the static variable initialization trick
C10_API bool LogAPIUsageFakeReturn(const std::string& context);
} // namespace detail

// Initializes the c10 logger.
C10_API void initLogging();

class C10_API LogState {
 public:
  // Enable logs under the given component alias to be emitted if they are
  // greater than or equal to the given level
  void enable_log(const char* log_qname, int64_t log_level);
  void enable_log(const std::string& log_qname, int64_t log_level);

  // Check if the given component alias is enabled with a log level less than or
  // equal to the given level
  bool is_log_enabled(const char* log_qname, int64_t log_level);
  bool is_log_enabled(const std::string& log_qname, int64_t log_level);

  ~LogState();

 private:
  struct CharPtrComparator {
    bool operator()(const char* a, const char* b) const {
      return std::strcmp(a, b) < 0;
    }
  };

  using LogStateMap = std::map<const char*, int64_t, CharPtrComparator>;

  LogStateMap log_qname_to_level;
};

C10_API LogState* getLogState();

class C10_API Log {
 public:
  Log(const char* log_qname,
      int64_t py_log_level,
      const SourceLocation& source_location,
      std::string msg);

  Log(const char* log_qname,
      int64_t py_log_level,
      const SourceLocation& source_location,
      const char* msg);

  Log(const char* log_qname,
      int64_t py_log_level,
      const SourceLocation& source_location,
      ::c10::detail::CompileTimeEmptyString msg);

  const std::string& log_qname() const;
  int64_t py_log_level() const;
  const SourceLocation& source_location() const;
  const std::string& msg() const;

 private:
  // The name of the component under which to emit the log message.
  std::string log_qname_;

  // The Python log level to emit the log as.
  int64_t py_log_level_;

  // Where the log happened.
  SourceLocation source_location_;

  // The actual log message.
  std::string msg_;
};

// Issue a log with a given message. Dispatched to the current
// log handler.
C10_API void log(const Log& log);

class C10_API LogHandler {
 public:
  virtual ~LogHandler() = default;

  virtual void process(const Log& log);
};

namespace LogUtils {

/// Sets the global log handler. This is not thread-safe, so it should
/// generally be called once during initialization or while holding the GIL
/// for programs that use python.
/// User is responsible for keeping the LogHandler alive until
/// it is not needed.
C10_API void set_log_handler(LogHandler* handler) noexcept(true);

/// Gets the global log handler.
C10_API LogHandler* get_log_handler() noexcept(true);

class C10_API LogHandlerGuard {
  LogHandler* prev_handler_;

 public:
  LogHandlerGuard(LogHandler* new_handler)
      : prev_handler_(c10::LogUtils::get_log_handler()) {
    c10::LogUtils::set_log_handler(new_handler);
  }
  ~LogHandlerGuard() {
    c10::LogUtils::set_log_handler(prev_handler_);
  }
};

} // namespace LogUtils

} // namespace c10

// TODO: Combine `TORCH_LOG` with `LOG_AT_FILE_LINE`
#ifdef DISABLE_LOG
#define TORCH_LOG(...) ((void)0);
#else
#define TORCH_LOG(log_qname, log_level, ...)                        \
  if (::c10::getLogState()->is_log_enabled(log_qname, log_level)) { \
    ::c10::log(::c10::Log(                                          \
        log_qname,                                                  \
        log_level,                                                  \
        {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},      \
        WARNING_MESSAGE_STRING(__VA_ARGS__)));                      \
  }
#endif

#endif // C10_UTIL_LOGGING_H_
