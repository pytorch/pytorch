#ifndef C10_UTIL_LOGGING_IS_NOT_GOOGLE_GLOG_H_
#define C10_UTIL_LOGGING_IS_NOT_GOOGLE_GLOG_H_

#include <chrono>
#include <climits>
#include <ctime>
#include <iomanip>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <c10/util/Flags.h>
#include <c10/util/logging_common.h>

const char CAFFE2_SEVERITY_PREFIX[] = "FEWIV";

namespace c10 {

// Log severity level constants.
const int GLOG_FATAL = 3;
const int GLOG_ERROR = 2;
const int GLOG_WARNING = 1;
const int GLOG_INFO = 0;

// Helpers for TORCH_CHECK_NOTNULL(). Two are necessary to support both raw
// pointers and smart pointers.
template <typename T>
T& CheckNotNullCommon(
    const char* file,
    int line,
    const char* names,
    T& t,
    bool fatal) {
  if (t == nullptr) {
    MessageLogger(file, line, GLOG_FATAL, fatal).stream()
        << "Check failed: '" << names << "' must be non NULL. ";
  }
  return t;
}

template <typename T>
T* CheckNotNull(
    const char* file,
    int line,
    const char* names,
    T* t,
    bool fatal) {
  return CheckNotNullCommon(file, line, names, t, fatal);
}

template <typename T>
T& CheckNotNull(
    const char* file,
    int line,
    const char* names,
    T& t,
    bool fatal) {
  return CheckNotNullCommon(file, line, names, t, fatal);
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

inline std::ostream& operator<<(
    std::ostream& out,
    const std::nullptr_t& /*unused*/) {
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
