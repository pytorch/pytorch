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

#include "c10/util/Flags.h"

// Log severity level constants.
const int FATAL = 3;
#if !defined(_MSC_VER) || !defined(ERROR)
// Windows defines the ERROR macro already, and as a result we will
// simply use that one. The downside is that one will now mix LOG(INFO)
// and LOG(ERROR) because ERROR is defined to be zero. Anyway, the
// recommended way is to use glog so fixing this is a low-pri item.
const int ERROR = 2;
#endif
const int WARNING = 1;
const int INFO = 0;
const char CAFFE2_SEVERITY_PREFIX[] = "FEWIV";

namespace c10 {
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
  MessageLogger(file, line, FATAL).stream() << message;
}

// Helpers for CHECK_NOTNULL(). Two are necessary to support both raw pointers
// and smart pointers.
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
    CAFFE2_LOG_THRESHOLD <= FATAL,
    "CAFFE2_LOG_THRESHOLD should at most be FATAL.");
// If n is under the compile time caffe log threshold, The _CAFFE_LOG(n)
// should not generate anything in optimized code.
#define LOG(n)                   \
  if (n >= CAFFE2_LOG_THRESHOLD) \
  ::c10::MessageLogger((char*)__FILE__, __LINE__, n).stream()
#define VLOG(n) LOG((-n))

#define LOG_IF(n, condition)                    \
  if (n >= CAFFE2_LOG_THRESHOLD && (condition)) \
  ::c10::MessageLogger((char*)__FILE__, __LINE__, n).stream()
#define VLOG_IF(n, condition) LOG_IF((-n), (condition))

#define VLOG_IS_ON(verboselevel) (CAFFE2_LOG_THRESHOLD <= -(verboselevel))

// Log only if condition is met.  Otherwise evaluates to void.
#define FATAL_IF(condition)            \
  condition ? (void)0                  \
            : ::c10::LoggerVoidify() & \
          ::c10::MessageLogger((char*)__FILE__, __LINE__, FATAL).stream()

// Check for a given boolean condition.
#define CHECK(condition) FATAL_IF(condition) << "Check failed: " #condition " "

#ifndef NDEBUG
// Debug only version of CHECK
#define DCHECK(condition) FATAL_IF(condition) << "Check failed: " #condition " "
#else
// Optimized version - generates no code.
#define DCHECK(condition) \
  while (false)           \
  CHECK(condition)
#endif // NDEBUG

#define CHECK_OP(val1, val2, op)                      \
  FATAL_IF(((val1) op (val2)))                        \
    << "Check failed: " #val1 " " #op " " #val2 " ("  \
    << (val1) << " vs. " << (val2) << ") "

// Check_op macro definitions
#define CHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define CHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define CHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define CHECK_LT(val1, val2) CHECK_OP(val1, val2, <)
#define CHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define CHECK_GT(val1, val2) CHECK_OP(val1, val2, >)

#ifndef NDEBUG
// Debug only versions of CHECK_OP macros.
#define DCHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define DCHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define DCHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define DCHECK_LT(val1, val2) CHECK_OP(val1, val2, <)
#define DCHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define DCHECK_GT(val1, val2) CHECK_OP(val1, val2, >)
#else // !NDEBUG
// These versions generate no code in optimized mode.
#define DCHECK_EQ(val1, val2) \
  while (false)               \
  CHECK_OP(val1, val2, ==)
#define DCHECK_NE(val1, val2) \
  while (false)               \
  CHECK_OP(val1, val2, !=)
#define DCHECK_LE(val1, val2) \
  while (false)               \
  CHECK_OP(val1, val2, <=)
#define DCHECK_LT(val1, val2) \
  while (false)               \
  CHECK_OP(val1, val2, <)
#define DCHECK_GE(val1, val2) \
  while (false)               \
  CHECK_OP(val1, val2, >=)
#define DCHECK_GT(val1, val2) \
  while (false)               \
  CHECK_OP(val1, val2, >)
#endif // NDEBUG

// Check that a pointer is not null.
#define CHECK_NOTNULL(val) \
  ::c10::CheckNotNull(     \
      __FILE__, __LINE__, "Check failed: '" #val "' Must be non NULL", (val))

#ifndef NDEBUG
// Debug only version of CHECK_NOTNULL
#define DCHECK_NOTNULL(val) \
  ::c10::CheckNotNull(      \
      __FILE__, __LINE__, "Check failed: '" #val "' Must be non NULL", (val))
#else // !NDEBUG
// Optimized version - generates no code.
#define DCHECK_NOTNULL(val) \
  while (false)             \
  CHECK_NOTNULL(val)
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
    std::ostream& out, const std::nullptr_t&) {
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
