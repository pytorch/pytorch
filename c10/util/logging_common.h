#ifndef C10_UTIL_LOGGING_COMMON_H_
#define C10_UTIL_LOGGING_COMMON_H_

#include <c10/macros/Export.h>
#include <sstream>

namespace c10 {

// MessageLogger that throws exceptions instead of aborting (glog version)
// or logs and may abort (non-glog version).
class C10_API MessageLogger {
 public:
  MessageLogger(
      const char* file,
      int line,
      int severity,
      bool exit_on_fatal = true);
  ~MessageLogger() noexcept(false);

  // Return the stream associated with the logger object.
  std::stringstream& stream();

 private:
  // When there is a fatal log, and fatal == true, we abort
  // otherwise, we throw.
  void DealWithFatal();

#if defined(ANDROID) && !defined(C10_USE_GLOG)
  const char* tag_{"native"};
#endif
  std::stringstream stream_;
  int severity_;
  bool exit_on_fatal_;
};

// This class is used to explicitly ignore values in the conditional
// logging macros. This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class C10_API LoggerVoidify {
 public:
  LoggerVoidify() = default;
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  void operator&(const std::ostream& s [[maybe_unused]]) {}
};

// Forward declarations for CheckNotNull functions
template <typename T>
T& CheckNotNullCommon(
    const char* file,
    int line,
    const char* names,
    T& t,
    bool fatal = true);

template <typename T>
T* CheckNotNull(
    const char* file,
    int line,
    const char* names,
    T* t,
    bool fatal = true);

template <typename T>
T& CheckNotNull(
    const char* file,
    int line,
    const char* names,
    T& t,
    bool fatal = true);

} // namespace c10

#endif // C10_UTIL_LOGGING_COMMON_H_
