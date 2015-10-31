#ifndef CAFFE2_CORE_LOGGING_IS_NOT_GOOGLE_GLOG_H_
#define CAFFE2_CORE_LOGGING_IS_NOT_GOOGLE_GLOG_H_

#ifdef ANDROID
#include <android/log.h>
#endif  // ANDROID

#include <algorithm>
#include <chrono>
#include <climits>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <set>
#include <sstream>
#ifdef CAFFE2_THROW_ON_FATAL
#include <stdexcept>
#endif  // CAFFE2_THROW_ON_FATAL
#include <vector>

#include "caffe2/core/flags.h"
CAFFE2_DECLARE_int(caffe2_log_level);

// Log severity level constants.
const int CAFFE_FATAL   = 3;
const int CAFFE_ERROR   = 2;
const int CAFFE_WARNING = 1;
const int CAFFE_INFO    = 0;
const char CAFFE_SEVERITY_PREFIX[] = "FEWIV";

namespace caffe2 {
class MessageLogger {
 public:
  MessageLogger(const char *file, int line, int severity)
    : severity_(severity) {
    if (severity_ < FLAGS_caffe2_log_level) {
      // Nothing needs to be logged.
      return;
    }
#ifdef ANDROID
    tag_ = "native";
#else  // !ANDROID
    tag_ = "";
#endif  // ANDROID
    // Pre-pend the stream with the file and line number.
    StripBasename(std::string(file), &filename_only_);
    /*
    time_t rawtime;
    struct tm * timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    std::chrono::nanoseconds ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch());
    */
    stream_ << "["
            << CAFFE_SEVERITY_PREFIX[std::min(4, CAFFE_FATAL - severity_)]
            //<< (timeinfo->tm_mon + 1) * 100 + timeinfo->tm_mday
            //<< std::setfill('0')
            //<< " " << std::setw(2) << timeinfo->tm_hour
            //<< ":" << std::setw(2) << timeinfo->tm_min
            //<< ":" << std::setw(2) << timeinfo->tm_sec
            //<< "." << std::setw(9) << ns.count() % 1000000000
            << " " << filename_only_ << ":" << line << "] ";
  }

  // Output the contents of the stream to the proper channel on destruction.
  ~MessageLogger() {
    if (severity_ < FLAGS_caffe2_log_level) {
      // Nothing needs to be logged.
      return;
    }
    stream_ << "\n";
#ifdef ANDROID
    static const int android_log_levels[] = {
        ANDROID_LOG_FATAL,    // CAFFE_LOG_FATAL
        ANDROID_LOG_ERROR,    // CAFFE_LOG_ERROR
        ANDROID_LOG_WARN,     // CAFFE_LOG_WARNING
        ANDROID_LOG_INFO,     // CAFFE_LOG_INFO
        ANDROID_LOG_DEBUG,    // CAFFE_VLOG(1)
        ANDROID_LOG_VERBOSE,  // CAFFE_VLOG(2) .. CAFFE_VLOG(N)
    };
    int android_level_index = CAFFE_FATAL - std::min(CAFFE_FATAL, severity_);
    int level = android_log_levels[std::min(android_level_index, 5)];
    // Output the log string the Android log at the appropriate level.
    __android_log_print(level, tag_, stream_.str().c_str());
    // Indicate termination if needed.
    if (severity_ == CAFFE_FATAL) {
      __android_log_print(ANDROID_LOG_FATAL, tag_.c_str(), "terminating.\n");
    }
#else  // !ANDROID
    if (severity_ >= FLAGS_caffe2_log_level) {
      // If not building on Android, log all output to std::cerr.
      std::cerr << stream_.str();
    }
#endif  // ANDROID
    if (severity_ == CAFFE_FATAL) {
      DealWithFatal();
    }
  }

  // Return the stream associated with the logger object.
  std::stringstream &stream() { return stream_; }

 private:
  void StripBasename(const std::string &full_path, std::string *filename) {
    const char kSeparator = '/';
    size_t pos = full_path.rfind(kSeparator);
    if (pos != std::string::npos) {
      *filename = full_path.substr(pos + 1, std::string::npos);
    } else {
      *filename = full_path;
    }
  }

#ifdef CAFFE2_THROW_ON_FATAL
  void DealWithFatal() {
    throw std::runtime_error("Caffe2 threw a log fatal.");
  }
#else
  // When there is a fatal log, we simply abort.
  void DealWithFatal() {
    abort();
  }
#endif

  std::string filename_only_;
  const char* tag_;
  std::stringstream stream_;
  int severity_;
};

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LoggerVoidify {
 public:
  LoggerVoidify() { }
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  void operator&(const std::ostream &s) { }
};

// Log a message and terminate.
template<class T>
void LogMessageFatal(const char *file, int line, const T &message) {
  MessageLogger(file, line, CAFFE_FATAL).stream() << message;
}

// Helpers for CAFFE_CHECK_NOTNULL(). Two are necessary to support both raw pointers
// and smart pointers.
template <typename T>
T& CheckNotNullCommon(const char *file, int line, const char *names, T& t) {
  if (t == nullptr) {
    LogMessageFatal(file, line, std::string(names));
  }
  return t;
}

template <typename T>
T* CheckNotNull(const char *file, int line, const char *names, T* t) {
  return CheckNotNullCommon(file, line, names, t);
}

template <typename T>
T& CheckNotNull(const char *file, int line, const char *names, T& t) {
  return CheckNotNullCommon(file, line, names, t);
}
}  // namespace caffe2

// ---------------------- Logging Macro definitions --------------------------


static_assert(CAFFE2_LOG_THRESHOLD <= CAFFE_FATAL,
              "CAFFE2_LOG_THRESHOLD should at most be FATAL.");
// If n is under the compile time caffe log threshold, The _CAFFE_LOG(n)
// should not generate anything in optimized code.
#define _CAFFE_LOG(n) \
  if (n >= CAFFE2_LOG_THRESHOLD) \
    ::caffe2::MessageLogger((char*)__FILE__, __LINE__, n).stream()
#define CAFFE_LOG_INFO _CAFFE_LOG(CAFFE_INFO)
#define CAFFE_LOG_WARNING _CAFFE_LOG(CAFFE_WARNING)
#define CAFFE_LOG_ERROR _CAFFE_LOG(CAFFE_ERROR)
#define CAFFE_LOG_FATAL _CAFFE_LOG(CAFFE_FATAL)
#define CAFFE_VLOG(n) _CAFFE_LOG((-n))

// Log only if condition is met.  Otherwise evaluates to void.
#define CAFFE_FATAL_IF(condition) \
  condition ? (void) 0 : ::caffe2::LoggerVoidify() & \
      ::caffe2::MessageLogger((char*)__FILE__, __LINE__, CAFFE_FATAL).stream()

// Check for a given boolean condition.
#define CAFFE_CHECK(condition) CAFFE_FATAL_IF(condition) \
        << "Check failed: " #condition " "

#ifndef NDEBUG
// Debug only version of CHECK
#define CAFFE_DCHECK(condition) CAFFE_FATAL_IF(condition) \
        << "Check failed: " #condition " "
#else
// Optimized version - generates no code.
#define CAFFE_DCHECK(condition) if(false) CAFFE_CHECK(condition)
#endif  // NDEBUG

#define CAFFE_CHECK_OP(val1, val2, op) CAFFE_FATAL_IF((val1 op val2)) \
  << "Check failed: " #val1 " " #op " " #val2 " "

// Check_op macro definitions
#define CAFFE_CHECK_EQ(val1, val2) CAFFE_CHECK_OP(val1, val2, ==)
#define CAFFE_CHECK_NE(val1, val2) CAFFE_CHECK_OP(val1, val2, !=)
#define CAFFE_CHECK_LE(val1, val2) CAFFE_CHECK_OP(val1, val2, <=)
#define CAFFE_CHECK_LT(val1, val2) CAFFE_CHECK_OP(val1, val2, <)
#define CAFFE_CHECK_GE(val1, val2) CAFFE_CHECK_OP(val1, val2, >=)
#define CAFFE_CHECK_GT(val1, val2) CAFFE_CHECK_OP(val1, val2, >)

#ifndef NDEBUG
// Debug only versions of CAFFE_CHECK_OP macros.
#define CAFFE_DCHECK_EQ(val1, val2) CAFFE_CHECK_OP(val1, val2, ==)
#define CAFFE_DCHECK_NE(val1, val2) CAFFE_CHECK_OP(val1, val2, !=)
#define CAFFE_DCHECK_LE(val1, val2) CAFFE_CHECK_OP(val1, val2, <=)
#define CAFFE_DCHECK_LT(val1, val2) CAFFE_CHECK_OP(val1, val2, <)
#define CAFFE_DCHECK_GE(val1, val2) CAFFE_CHECK_OP(val1, val2, >=)
#define CAFFE_DCHECK_GT(val1, val2) CAFFE_CHECK_OP(val1, val2, >)
#else  // !NDEBUG
// These versions generate no code in optimized mode.
#define CAFFE_DCHECK_EQ(val1, val2) if(false) CAFFE_CHECK_OP(val1, val2, ==)
#define CAFFE_DCHECK_NE(val1, val2) if(false) CAFFE_CHECK_OP(val1, val2, !=)
#define CAFFE_DCHECK_LE(val1, val2) if(false) CAFFE_CHECK_OP(val1, val2, <=)
#define CAFFE_DCHECK_LT(val1, val2) if(false) CAFFE_CHECK_OP(val1, val2, <)
#define CAFFE_DCHECK_GE(val1, val2) if(false) CAFFE_CHECK_OP(val1, val2, >=)
#define CAFFE_DCHECK_GT(val1, val2) if(false) CAFFE_CHECK_OP(val1, val2, >)
#endif  // NDEBUG

// Check that a pointer is not null.
#define CAFFE_CHECK_NOTNULL(val) \
  ::caffe2::CheckNotNull( \
      __FILE__, __LINE__, "Check failed: '" #val "' Must be non NULL", (val))

#ifndef NDEBUG
// Debug only version of CAFFE_CHECK_NOTNULL
#define CAFFE_DCHECK_NOTNULL(val) \
  ::caffe2::CheckNotNull( \
      __FILE__, __LINE__, "Check failed: '" #val "' Must be non NULL", (val))
#else  // !NDEBUG
// Optimized version - generates no code.
#define CAFFE_DCHECK_NOTNULL(val) if (false) CAFFE_CHECK_NOTNULL(val)
#endif  // NDEBUG

#endif  // CAFFE2_CORE_LOGGING_IS_NOT_GOOGLE_GLOG_H_
