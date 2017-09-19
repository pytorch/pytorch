#ifndef CAFFE2_CORE_LOGGING_H_
#define CAFFE2_CORE_LOGGING_H_

#include <climits>
#include <exception>
#include <functional>
#include <limits>
#include <sstream>

#include "caffe2/core/flags.h"
#include "caffe2/proto/caffe2.pb.h"

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

/**
 * A utility to allow one to show log info to stderr after the program starts.
 *
 * This is similar to calling GLOG's --logtostderr, or setting caffe2_log_level
 * to smaller than INFO. You are recommended to only use this in a few sparse
 * cases, such as when you want to write a tutorial or something. Normally, use
 * the commandline flags to set the log level.
 */
void ShowLogInfoToStderr();

inline void MakeStringInternal(std::stringstream& /*ss*/) {}

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

template <class Container>
inline string Join(const string& delimiter, const Container& v) {
  std::stringstream s;
  int cnt = static_cast<int64_t>(v.size()) - 1;
  for (auto i = v.begin(); i != v.end(); ++i, --cnt) {
    s << (*i) << (cnt ? delimiter : "");
  }
  return s.str();
}

// Obtains the base name from a full path.
string StripBasename(const std::string& full_path);

// Replace all occurrences of "from" substring to "to" string.
// Returns number of replacements
size_t ReplaceAll(string& s, const char* from, const char* to);

void SetStackTraceFetcher(std::function<string(void)> fetcher);

void SetOperatorLogger(std::function<void(const OperatorDef&)> tracer);
std::function<void(const OperatorDef&)> GetOperatorLogger();

class EnforceNotMet : public std::exception {
 public:
  EnforceNotMet(
      const char* file,
      const int line,
      const char* condition,
      const string& msg,
      const void* caller=nullptr);
  void AppendMessage(const string& msg);
  string msg() const;
  inline const vector<string>& msg_stack() const {
    return msg_stack_;
  }

  const char* what() const noexcept override;

  const void* caller() const noexcept;

 private:
  vector<string> msg_stack_;
  string full_msg_;
  string stack_trace_;
  const void* caller_;
};

#define CAFFE_ENFORCE(condition, ...)                                         \
  do {                                                                        \
    if (!(condition)) {                                                       \
      throw ::caffe2::EnforceNotMet(                                          \
          __FILE__, __LINE__, #condition, ::caffe2::MakeString(__VA_ARGS__)); \
    }                                                                         \
  } while (false)

#define CAFFE_ENFORCE_WITH_CALLER(condition, ...)                             \
  do {                                                                        \
    if (!(condition)) {                                                       \
      throw ::caffe2::EnforceNotMet(                                          \
          __FILE__, __LINE__, #condition, ::caffe2::MakeString(__VA_ARGS__), this); \
    }                                                                         \
  } while (false)

#define CAFFE_THROW(...)         \
  throw ::caffe2::EnforceNotMet( \
      __FILE__, __LINE__, "", ::caffe2::MakeString(__VA_ARGS__))

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
 *   inline EnforceFailMessage IsVector(const vector<TIndex>& shape) {
 *     if (shape.size() == 1) { return EnforceOK(); }
 *     return MakeString("Shape ", shape, " is not a vector");
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

struct EnforceOK {};

class EnforceFailMessage {
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

  /* implicit */ EnforceFailMessage(std::string&& msg) {
    msg_ = new std::string(std::move(msg));
  }
  inline bool bad() const {
    return msg_ != nullptr;
  }
  std::string get_message_and_free(std::string&& extra) const {
    std::string r;
    if (extra.empty()) {
      r = std::move(*msg_);
    } else {
      r = ::caffe2::MakeString(std::move(*msg_), ". ", std::move(extra));
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
    return MakeString(x, " vs ", y);                         \
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
    using namespace ::caffe2::enforce_detail;                           \
    const EnforceFailMessage& CAFFE_ENFORCE_THAT_IMPL_r_ = (condition); \
    if (CAFFE_ENFORCE_THAT_IMPL_r_.bad()) {                             \
      throw ::caffe2::EnforceNotMet(                                    \
          __FILE__,                                                     \
          __LINE__,                                                     \
          expr,                                                         \
          CAFFE_ENFORCE_THAT_IMPL_r_.get_message_and_free(              \
              ::caffe2::MakeString(__VA_ARGS__)));                      \
    }                                                                   \
  } while (false)

#define CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(condition, expr, ...)      \
  do {                                                                 \
    using namespace ::caffe2::enforce_detail;                          \
    const EnforceFailMessage& CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER_r_ = \
        (condition);                                                   \
    if (CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER_r_.bad()) {                \
      throw ::caffe2::EnforceNotMet(                                   \
          __FILE__,                                                    \
          __LINE__,                                                    \
          expr,                                                        \
          CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER_r_.get_message_and_free( \
              ::caffe2::MakeString(__VA_ARGS__)),                      \
          this);                                                       \
    }                                                                  \
  } while (false)
}

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
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(Equals((x), (y)), #x " == " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_NE_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(NotEquals((x), (y)), #x " != " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_LE_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(LessEquals((x), (y)), #x " <= " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_LT_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(Less((x), (y)), #x " < " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_GE_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(GreaterEquals((x), (y)), #x " >= " #y, __VA_ARGS__)
#define CAFFE_ENFORCE_GT_WITH_CALLER(x, y, ...) \
  CAFFE_ENFORCE_THAT_IMPL_WITH_CALLER(Greater((x), (y)), #x " > " #y, __VA_ARGS__)
} // namespace caffe2

#endif // CAFFE2_CORE_LOGGING_H_
