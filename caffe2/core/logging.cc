#include "caffe2/core/logging.h"
#include "caffe2/core/flags.h"

#include <algorithm>
#include <cstring>
#include <numeric>

// Common code that we use regardless of whether we use glog or not.

CAFFE2_DEFINE_bool(caffe2_use_fatal_for_enforce, false,
                   "If set true, when CAFFE_ENFORCE is not met, abort instead "
                   "of throwing an exception.");

namespace caffe2 {
std::string StripBasename(const std::string &full_path) {
  const char kSeparator = '/';
  size_t pos = full_path.rfind(kSeparator);
  if (pos != std::string::npos) {
    return full_path.substr(pos + 1, std::string::npos);
  } else {
    return full_path;
  }
}

size_t ReplaceAll(string& s, const char* from, const char* to) {
  CAFFE_ENFORCE(from && *from);
  CAFFE_ENFORCE(to);

  size_t numReplaced = 0;
  string::size_type lenFrom = std::strlen(from);
  string::size_type lenTo = std::strlen(to);
  for (string::size_type pos = s.find(from); pos != string::npos;
       pos = s.find(from, pos + lenTo)) {
    s.replace(pos, lenFrom, to);
    numReplaced++;
  }
  return numReplaced;
}

static std::function<string(void)> FetchStackTrace = []() { return ""; };

void SetStackTraceFetcher(std::function<string(void)> fetcher) {
  FetchStackTrace = fetcher;
}

static std::function<void(const OperatorDef&)> OperatorLogger =
    [](const OperatorDef&) { return; };

void SetOperatorLogger(std::function<void(const OperatorDef&)> tracer) {
  OperatorLogger = tracer;
}

std::function<void(const OperatorDef&)> GetOperatorLogger() {
  return OperatorLogger;
}

EnforceNotMet::EnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const string& msg,
    const void* caller)
    : msg_stack_{MakeString(
          "[enforce fail at ",
          StripBasename(std::string(file)),
          ":",
          line,
          "] ",
          condition,
          ". ",
          msg,
          " ")},
      stack_trace_(FetchStackTrace()) {
  if (FLAGS_caffe2_use_fatal_for_enforce) {
    LOG(FATAL) << msg_stack_[0];
  }
  caller_ = caller;
  full_msg_ = this->msg();
}

void EnforceNotMet::AppendMessage(const string& msg) {
  msg_stack_.push_back(msg);
  full_msg_ = this->msg();
}

string EnforceNotMet::msg() const {
  return std::accumulate(msg_stack_.begin(), msg_stack_.end(), string("")) +
      stack_trace_;
}

const char* EnforceNotMet::what() const noexcept {
  return full_msg_.c_str();
}

const void* EnforceNotMet::caller() const noexcept {
  return caller_;
}

}  // namespace caffe2


#ifdef CAFFE2_USE_GOOGLE_GLOG

#ifdef CAFFE2_USE_GFLAGS
// GLOG's minloglevel
CAFFE2_DECLARE_int(minloglevel);
// GLOG's verbose log value.
CAFFE2_DECLARE_int(v);
// GLOG's logtostderr value
CAFFE2_DECLARE_bool(logtostderr);

#else

using fLI::FLAGS_minloglevel;
using fLI::FLAGS_v;
using fLB::FLAGS_logtostderr;

#endif // CAFFE2_USE_GFLAGS

CAFFE2_DEFINE_int(caffe2_log_level, google::GLOG_ERROR,
                  "The minimum log level that caffe2 will output.");

// Google glog's api does not have an external function that allows one to check
// if glog is initialized or not. It does have an internal function - so we are
// declaring it here. This is a hack but has been used by a bunch of others too
// (e.g. Torch).
namespace google {
namespace glog_internal_namespace_ {
bool IsGoogleLoggingInitialized();
}  // namespace glog_internal_namespace_
}  // namespace google


namespace caffe2 {
bool InitCaffeLogging(int* argc, char** argv) {
  if (*argc == 0) return true;
#if !defined(_MSC_VER)
  // This trick can only be used on UNIX platforms
  if (!::google::glog_internal_namespace_::IsGoogleLoggingInitialized())
#endif
  {
    ::google::InitGoogleLogging(argv[0]);
#if !defined(_MSC_VER)
  // This is never defined on Windows
    ::google::InstallFailureSignalHandler();
#endif
  }
  // If caffe2_log_level is set and is lower than the min log level by glog,
  // we will transfer the caffe2_log_level setting to glog to override that.
  FLAGS_minloglevel = std::min(FLAGS_caffe2_log_level, FLAGS_minloglevel);
  // If caffe2_log_level is explicitly set, let's also turn on logtostderr.
  if (FLAGS_caffe2_log_level < google::GLOG_ERROR) {
    FLAGS_logtostderr = 1;
  }
  // Also, transfer the caffe2_log_level verbose setting to glog.
  if (FLAGS_caffe2_log_level < 0) {
    FLAGS_v = std::min(FLAGS_v, -FLAGS_caffe2_log_level);
  }
  return true;
}

void ShowLogInfoToStderr() {
  FLAGS_logtostderr = 1;
  FLAGS_minloglevel = std::min(FLAGS_minloglevel, google::GLOG_INFO);
}
}  // namespace caffe2

#else  // !CAFFE2_USE_GOOGLE_GLOG

#ifdef ANDROID
#include <android/log.h>
#endif // ANDROID

CAFFE2_DEFINE_int(caffe2_log_level, ERROR,
                  "The minimum log level that caffe2 will output.");

namespace caffe2 {
bool InitCaffeLogging(int* argc, char** argv) {
  // When doing InitCaffeLogging, we will assume that caffe's flag paser has
  // already finished.
  if (*argc == 0) return true;
  if (!CommandLineFlagsHasBeenParsed()) {
    std::cerr << "InitCaffeLogging() has to be called after "
                 "ParseCaffeCommandLineFlags. Modify your program to make sure "
                 "of this." << std::endl;
    return false;
  }
  if (FLAGS_caffe2_log_level > FATAL) {
    std::cerr << "The log level of Caffe2 has to be no larger than FATAL("
              << FATAL << "). Capping it to FATAL." << std::endl;
    FLAGS_caffe2_log_level = FATAL;
  }
  return true;
}

void ShowLogInfoToStderr() {
  FLAGS_caffe2_log_level = INFO;
}

MessageLogger::MessageLogger(const char *file, int line, int severity)
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
  /*
  time_t rawtime;
  struct tm * timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  std::chrono::nanoseconds ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch());
  */
  stream_ << "[" << CAFFE2_SEVERITY_PREFIX[std::min(4, FATAL - severity_)]
          //<< (timeinfo->tm_mon + 1) * 100 + timeinfo->tm_mday
          //<< std::setfill('0')
          //<< " " << std::setw(2) << timeinfo->tm_hour
          //<< ":" << std::setw(2) << timeinfo->tm_min
          //<< ":" << std::setw(2) << timeinfo->tm_sec
          //<< "." << std::setw(9) << ns.count() % 1000000000
          << " " << StripBasename(std::string(file)) << ":" << line << "] ";
}

// Output the contents of the stream to the proper channel on destruction.
MessageLogger::~MessageLogger() {
  if (severity_ < FLAGS_caffe2_log_level) {
    // Nothing needs to be logged.
    return;
  }
  stream_ << "\n";
#ifdef ANDROID
  static const int android_log_levels[] = {
      ANDROID_LOG_FATAL,    // LOG_FATAL
      ANDROID_LOG_ERROR,    // LOG_ERROR
      ANDROID_LOG_WARN,     // LOG_WARNING
      ANDROID_LOG_INFO,     // LOG_INFO
      ANDROID_LOG_DEBUG,    // VLOG(1)
      ANDROID_LOG_VERBOSE,  // VLOG(2) .. VLOG(N)
  };
  int android_level_index = FATAL - std::min(FATAL, severity_);
  int level = android_log_levels[std::min(android_level_index, 5)];
  // Output the log string the Android log at the appropriate level.
  __android_log_print(level, tag_, "%s", stream_.str().c_str());
  // Indicate termination if needed.
  if (severity_ == FATAL) {
    __android_log_print(ANDROID_LOG_FATAL, tag_, "terminating.\n");
  }
#else  // !ANDROID
  if (severity_ >= FLAGS_caffe2_log_level) {
    // If not building on Android, log all output to std::cerr.
    std::cerr << stream_.str();
  }
#endif  // ANDROID
  if (severity_ == FATAL) {
    DealWithFatal();
  }
}

}  // namespace caffe2

#endif  // !CAFFE2_USE_GOOGLE_GLOG
