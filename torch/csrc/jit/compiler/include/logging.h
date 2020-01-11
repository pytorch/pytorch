#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace torch {
namespace jit {
namespace compiler {

// TODO: Switch the entire file to the PT version

const int FATAL = 3;
const int ERROR = 2;
const int WARNING = 1;
const int INFO = 0;

__attribute__((noreturn))
inline void assert_unreachable(const char *msg) {
  std::cerr << msg << "\n";
  std::abort();
}

template <int severity>
class MessageLogger {
 public:
  static std::string SeverityToString(int sev) {
    switch (sev) {
      case FATAL:
        return "FATAL";
      case ERROR:
        return "ERROR";
      case WARNING:
        return "WARNING";
      case INFO:
        return "INFO";
    }
    assert_unreachable("No such severity level");
  }

  MessageLogger(const char* file, int line)
      : severity_(severity) {
    stream_ << SeverityToString(severity) << ":" << file << ":" << line << ": ";
  }

  ~MessageLogger();

  // Return the stream associated with the logger object.
  std::stringstream& stream() {
    return stream_;
  }

 private:
  // When there is a fatal log, we simply abort.
__attribute__((noreturn))
  void DealWithFatal() {
    abort();
  }

  const char* tag_;
  std::stringstream stream_;
  int severity_;
};

class LoggerVoidify {
 public:
  LoggerVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  void operator&(const std::ostream& s) {}
};

template <int severity>
MessageLogger<severity>::~MessageLogger() {
  std::cerr << stream_.str() << std::flush;
}

template <>
__attribute__((noreturn))
inline MessageLogger<FATAL>::~MessageLogger() {
  std::cerr << stream_.str() << std::flush;
  DealWithFatal();
}
 
// Log a message and terminate.
template <class T>
void LogMessageFatal(const char* file, int line, const T& message) {
  MessageLogger<FATAL>(file, line).stream() << message;
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

#define LOG(n) MessageLogger<n>((char*)__FILE__, __LINE__).stream()

#define FATAL_IF(condition)     \
  condition ? (void)0           \
            : LoggerVoidify() & \
          MessageLogger<FATAL>((char*)__FILE__, __LINE__).stream()

#define CHECK(condition) \
  FATAL_IF(condition) << "Check failed: (" #condition ") "

#ifndef NDEBUG
// Debug only version of CHECK
#define DCHECK(condition) CHECK(condition)
#else
// Optimized version - generates no code.
#define DCHECK(condition) \
  while (false)           \
  CHECK(condition)
#endif // NDEBUG

#define CHECK_OP(val1, val2, op)                                            \
  FATAL_IF((val1 op val2)) << "Check failed: " #val1 " " #op " " #val2 ": " \
                           << (val1) << " vs " << (val2)

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

} // namespace compiler
} // namespace jit
} // namespace torch
