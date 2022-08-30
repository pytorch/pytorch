#ifndef C10_UTIL_LOGGING_IS_NOT_GOOGLE_GLOG_H_
#define C10_UTIL_LOGGING_IS_NOT_GOOGLE_GLOG_H_

#include <chrono>
#include <climits>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <c10/util/Flags.h>

const char CAFFE2_SEVERITY_PREFIX[] = "FEWIV";

namespace c10 {

// Log severity level constants.
const int GLOG_FATAL = 3;
const int GLOG_ERROR = 2;
const int GLOG_WARNING = 1;
const int GLOG_INFO = 0;

class C10_API MessageLogger {
 public:
  MessageLogger(const char* file, int line, int severity);
  ~MessageLogger();
  // Return the stream associated with the logger object.
  std::stringstream& stream() {
    return stream_;
  }

 private:
  // When there is a fatal log, we simply abort.
  void DealWithFatal() {
    abort();
  }

  const char* tag_;
  std::stringstream stream_;
  int severity_;
};

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class C10_API LoggerVoidify {
 public:
  LoggerVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  void operator&(const std::ostream& s) {}
};

// Log a message and terminate.
template <class T>
void LogMessageFatal(const char* file, int line, const T& message) {
  MessageLogger(file, line, GLOG_FATAL).stream() << message;
}

// Helpers for TORCH_CHECK_NOTNULL(). Two are necessary to support both raw
// pointers and smart pointers.
template <typename T>
T& CheckNotNullCommon(const char* file, int line, const char* names, T& t) {
  if (t == nullptr) {
    LogMessageFatal(file, line, std::string(names));
  }
  return t;
}

template <typename T>
T* CheckNotNull(const char* file, int line, const char* names, T* t) {
  return CheckNotNullCommon(file, line, names, t);
}

template <typename T>
T& CheckNotNull(const char* file, int line, const char* names, T& t) {
  return CheckNotNullCommon(file, line, names, t);
}
} // namespace c10

// ---------------------- Logging Macro definitions --------------------------

static_assert(
    CAFFE2_LOG_THRESHOLD <= ::c10::GLOG_FATAL,
    "CAFFE2_LOG_THRESHOLD should at most be GLOG_FATAL.");
// If n is under the compile time caffe log threshold, The _CAFFE_LOG(n)
// should not generate anything in optimized code.
#define LOG(n)                                 \
  if (::c10::GLOG_##n >= CAFFE2_LOG_THRESHOLD) \
  ::c10::MessageLogger(__FILE__, __LINE__, ::c10::GLOG_##n).stream()
#define VLOG(n)                   \
  if (-n >= CAFFE2_LOG_THRESHOLD) \
  ::c10::MessageLogger(__FILE__, __LINE__, -n).stream()

#define LOG_IF(n, condition)                                  \
  if (::c10::GLOG_##n >= CAFFE2_LOG_THRESHOLD && (condition)) \
  ::c10::MessageLogger(__FILE__, __LINE__, ::c10::GLOG_##n).stream()
#define VLOG_IF(n, condition)                    \
  if (-n >= CAFFE2_LOG_THRESHOLD && (condition)) \
  ::c10::MessageLogger(__FILE__, __LINE__, -n).stream()

#define VLOG_IS_ON(verboselevel) (CAFFE2_LOG_THRESHOLD <= -(verboselevel))

// Log with source location information override (to be used in generic
// warning/error handlers implemented as functions, not macros)
#define LOG_AT_FILE_LINE(n, file, line)        \
  if (::c10::GLOG_##n >= CAFFE2_LOG_THRESHOLD) \
  ::c10::MessageLogger(file, line, ::c10::GLOG_##n).stream()

// Log only if condition is met.  Otherwise evaluates to void.
#define FATAL_IF(condition)            \
  condition ? (void)0                  \
            : ::c10::LoggerVoidify() & \
          ::c10::MessageLogger(__FILE__, __LINE__, ::c10::GLOG_FATAL).stream()

// Check for a given boolean condition.
#define CHECK(condition) FATAL_IF(condition) << "Check failed: " #condition " "

#ifndef NDEBUG
// Debug only version of CHECK
#define DCHECK(condition) FATAL_IF(condition) << "Check failed: " #condition " "
#define DLOG(severity) LOG(severity)
#else // NDEBUG
// Optimized version - generates no code.
#define DCHECK(condition) \
  while (false)           \
  CHECK(condition)

#define DLOG(n)                   \
  true ? (void)0                  \
       : ::c10::LoggerVoidify() & \
          ::c10::MessageLogger(__FILE__, __LINE__, ::c10::GLOG_##n).stream()
#endif // NDEBUG

#define TORCH_CHECK_OP(val1, val2, op)                                        \
  FATAL_IF(((val1)op(val2))) << "Check failed: " #val1 " " #op " " #val2 " (" \
                             << (val1) << " vs. " << (val2) << ") "

// TORCH_CHECK_OP macro definitions
#define TORCH_CHECK_EQ(val1, val2) TORCH_CHECK_OP(val1, val2, ==)
#define TORCH_CHECK_NE(val1, val2) TORCH_CHECK_OP(val1, val2, !=)
#define TORCH_CHECK_LE(val1, val2) TORCH_CHECK_OP(val1, val2, <=)
#define TORCH_CHECK_LT(val1, val2) TORCH_CHECK_OP(val1, val2, <)
#define TORCH_CHECK_GE(val1, val2) TORCH_CHECK_OP(val1, val2, >=)
#define TORCH_CHECK_GT(val1, val2) TORCH_CHECK_OP(val1, val2, >)

#ifndef NDEBUG
// Debug only versions of TORCH_CHECK_OP macros.
#define TORCH_DCHECK_EQ(val1, val2) TORCH_CHECK_OP(val1, val2, ==)
#define TORCH_DCHECK_NE(val1, val2) TORCH_CHECK_OP(val1, val2, !=)
#define TORCH_DCHECK_LE(val1, val2) TORCH_CHECK_OP(val1, val2, <=)
#define TORCH_DCHECK_LT(val1, val2) TORCH_CHECK_OP(val1, val2, <)
#define TORCH_DCHECK_GE(val1, val2) TORCH_CHECK_OP(val1, val2, >=)
#define TORCH_DCHECK_GT(val1, val2) TORCH_CHECK_OP(val1, val2, >)
#else // !NDEBUG
// These versions generate no code in optimized mode.
#define TORCH_DCHECK_EQ(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, ==)
#define TORCH_DCHECK_NE(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, !=)
#define TORCH_DCHECK_LE(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, <=)
#define TORCH_DCHECK_LT(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, <)
#define TORCH_DCHECK_GE(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, >=)
#define TORCH_DCHECK_GT(val1, val2) \
  while (false)                     \
  TORCH_CHECK_OP(val1, val2, >)
#endif // NDEBUG

// Check that a pointer is not null.
#define TORCH_CHECK_NOTNULL(val) \
  ::c10::CheckNotNull(           \
      __FILE__, __LINE__, "Check failed: '" #val "' Must be non NULL", (val))

#ifndef NDEBUG
// Debug only version of TORCH_CHECK_NOTNULL
#define TORCH_DCHECK_NOTNULL(val) \
  ::c10::CheckNotNull(            \
      __FILE__, __LINE__, "Check failed: '" #val "' Must be non NULL", (val))
#else // !NDEBUG
// Optimized version - generates no code.
#define TORCH_DCHECK_NOTNULL(val) \
  while (false)                   \
  TORCH_CHECK_NOTNULL(val)
#endif // NDEBUG

// ---------------------- Support for std objects --------------------------
// These are adapted from glog to support a limited set of logging capability
// for STL objects.

namespace std {
// Forward declare these two, and define them after all the container streams
// operators so that we can recurse from pair -> container -> container -> pair
// properly.
template <class First, class Second>
std::ostream& operator<<(std::ostream& out, const std::pair<First, Second>& p);
} // namespace std

namespace c10 {
template <class Iter>
void PrintSequence(std::ostream& ss, Iter begin, Iter end);
} // namespace c10

namespace std {
#define INSTANTIATE_FOR_CONTAINER(container)               \
  template <class... Types>                                \
  std::ostream& operator<<(                                \
      std::ostream& out, const container<Types...>& seq) { \
    c10::PrintSequence(out, seq.begin(), seq.end());       \
    return out;                                            \
  }

INSTANTIATE_FOR_CONTAINER(std::vector)
INSTANTIATE_FOR_CONTAINER(std::map)
INSTANTIATE_FOR_CONTAINER(std::set)
#undef INSTANTIATE_FOR_CONTAINER

template <class First, class Second>
inline std::ostream& operator<<(
    std::ostream& out,
    const std::pair<First, Second>& p) {
  out << '(' << p.first << ", " << p.second << ')';
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const std::nullptr_t&) {
  out << "(null)";
  return out;
}
} // namespace std

namespace c10 {
template <class Iter>
inline void PrintSequence(std::ostream& out, Iter begin, Iter end) {
  // Output at most 100 elements -- appropriate if used for logging.
  for (int i = 0; begin != end && i < 100; ++i, ++begin) {
    if (i > 0)
      out << ' ';
    out << *begin;
  }
  if (begin != end) {
    out << " ...";
  }
}
} // namespace c10

#endif // C10_UTIL_LOGGING_IS_NOT_GOOGLE_GLOG_H_
