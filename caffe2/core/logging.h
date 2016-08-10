#ifndef CAFFE2_CORE_LOGGING_H_
#define CAFFE2_CORE_LOGGING_H_

#include <climits>
#include <exception>
#include <sstream>

#include "caffe2/core/flags.h"

// CAFFE2_LOG_THRESHOLD is a compile time flag that would allow us to turn off
// logging at compile time so no logging message below that level is produced
// at all. The value should be between INT_MIN and CAFFE_FATAL.
#ifndef CAFFE2_LOG_THRESHOLD
// If we have not defined the compile time log threshold, we keep all the
// log cases.
#define CAFFE2_LOG_THRESHOLD INT_MIN
#endif // CAFFE2_LOG_THRESHOLD

// Below are different implementations for glog and non-glog cases.
#ifdef CAFFE2_USE_GOOGLE_GLOG
#include "caffe2/core/logging_is_google_glog.h"
#else // !CAFFE2_USE_GOOGLE_GLOG
#include "caffe2/core/logging_is_not_google_glog.h"
#endif // CAFFE2_USE_GOOGLE_GLOG

CAFFE2_DECLARE_int(caffe2_log_level);
CAFFE2_DECLARE_bool(caffe2_use_fatal_for_enforce);

namespace caffe2 {
// Functions that we use for initialization.
bool InitCaffeLogging(int* argc, char** argv);

constexpr bool IsUsingGoogleLogging() {
#ifdef CAFFE2_USE_GOOGLE_GLOG
  return true;
#else
  return false;
#endif
}

inline void MakeStringInternal(std::stringstream& ss) {}

template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) {
  ss << t;
}

template <typename T, typename... Args>
inline void
MakeStringInternal(std::stringstream& ss, const T& t, const Args&... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
string MakeString(const Args&... args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return string(ss.str());
}

// Specializations for already-a-string types.
template <>
inline string MakeString(const string& str) {
  return str;
}
inline string MakeString(const char* c_str) {
  return string(c_str);
}

// Obtains the base name from a full path.
string StripBasename(const std::string& full_path);

// Replace all occurrences of "from" substring to "to" string.
// Returns number of replacements
size_t ReplaceAll(string& s, const char* from, const char* to);

class EnforceNotMet : public std::exception {
 public:
  EnforceNotMet(
      const char* file,
      const int line,
      const char* condition,
      const string& msg);
  void AppendMessage(const string& msg);
  string msg() const;
  inline const vector<string>& msg_stack() const {
    return msg_stack_;
  }

 private:
  vector<string> msg_stack_;
};

// Exception handling: if we are enabling exceptions, we will turn on the caffe
// enforce capabilities. If one does not allow exceptions during compilation
// time, we will simply do LOG(FATAL).
#ifdef __EXCEPTIONS

#define CAFFE_ENFORCE(condition, ...)                                         \
  do {                                                                        \
    if (!(condition)) {                                                       \
      throw ::caffe2::EnforceNotMet(                                          \
          __FILE__, __LINE__, #condition, ::caffe2::MakeString(__VA_ARGS__)); \
    }                                                                         \
  } while (false)

#define CAFFE_THROW(...)         \
  throw ::caffe2::EnforceNotMet( \
      __FILE__, __LINE__, "", ::caffe2::MakeString(__VA_ARGS__))

#else // __EXCEPTIONS

#define CAFFE_ENFORCE(condition, ...)         \
  CHECK(condition) << "[exception as fatal] " \
                   << ::caffe2::MakeString(__VA_ARGS__)

#define CAFFE_THROW(...) \
  LOG(FATAL) << "[exception as fatal] " << ::caffe2::MakeString(__VA_ARGS__);

#endif // __EXCEPTIONS

#define CAFFE_FAIL(...) \
  static_assert(        \
      false, "CAFFE_FAIL is renamed CAFFE_THROW. Kindly change your code.")

} // namespace caffe2

#endif // CAFFE2_CORE_LOGGING_H_
