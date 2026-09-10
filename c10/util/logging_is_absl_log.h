#ifndef C10_UTIL_LOGGING_IS_ABSL_LOG_H_
#define C10_UTIL_LOGGING_IS_ABSL_LOG_H_

#include <iomanip>
#include <map>
#include <set>
#include <vector>

#include <absl/base/log_severity.h>
#include <absl/log/check.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>

#include <c10/util/logging_common.h>

namespace c10 {

template <typename T>
T& CheckNotNullCommon(
    const char* file,
    int line,
    const char* names,
    T& t,
    bool fatal) {
  if (t == nullptr) {
    MessageLogger(
        SourceLocation::current(file, nullptr, line),
        static_cast<int>(::absl::LogSeverity::kFatal),
        fatal)
            .stream()
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

// Log with source location information override (to be used in generic
// warning/error handlers implemented as functions, not macros)
#define LOG_AT_FILE_LINE(n, file, line) LOG(n).AtLocation(file, line)

// ---------------------- Support for std objects --------------------------
// These are adapted to support logging capability for STL objects with absl
// log.

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

#endif // C10_UTIL_LOGGING_IS_ABSL_LOG_H_
