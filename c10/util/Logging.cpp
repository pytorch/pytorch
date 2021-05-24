#include <c10/util/Backtrace.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#ifdef FBCODE_CAFFE2
#include <folly/synchronization/SanitizeThread.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>

// Common code that we use regardless of whether we use glog or not.

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_bool(
    caffe2_use_fatal_for_enforce,
    false,
    "If set true, when CAFFE_ENFORCE is not met, abort instead "
    "of throwing an exception.");

namespace c10 {

namespace {
// NOLINTNEXTLINE(modernize-redundant-void-arg)
std::function<string(void)>* GetFetchStackTrace() {
  static std::function<string(void)> func = []() {
    return get_backtrace(/*frames_to_skip=*/1);
  };
  return &func;
};
} // namespace

void SetStackTraceFetcher(std::function<string(void)> fetcher) {
  *GetFetchStackTrace() = fetcher;
}

void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller) {
  c10::Error e(file, line, condition, msg, (*GetFetchStackTrace())(), caller);
  if (FLAGS_caffe2_use_fatal_for_enforce) {
    LOG(FATAL) << e.msg();
  }
  throw e;
}

void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const char* msg,
    const void* caller) {
  ThrowEnforceNotMet(file, line, condition, std::string(msg), caller);
}

void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller) {
  throw c10::EnforceFiniteError(
      file, line, condition, msg, (*GetFetchStackTrace())(), caller);
}

void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    const char* condition,
    const char* msg,
    const void* caller) {
  ThrowEnforceFiniteNotMet(file, line, condition, std::string(msg), caller);
}
// PyTorch-style error message
// (This must be defined here for access to GetFetchStackTrace)
Error::Error(SourceLocation source_location, std::string msg)
    : Error(
          std::move(msg),
          str("Exception raised from ",
              source_location,
              " (most recent call first):\n",
              (*GetFetchStackTrace())())) {}

using APIUsageLoggerType = std::function<void(const std::string&)>;
using DDPUsageLoggerType = std::function<void(const DDPLoggingData&)>;

namespace {
bool IsAPIUsageDebugMode() {
  const char* val = getenv("PYTORCH_API_USAGE_STDERR");
  return val && *val; // any non-empty value
}

void APIUsageDebug(const string& event) {
  // use stderr to avoid messing with glog
  std::cerr << "PYTORCH_API_USAGE " << event << std::endl;
}

APIUsageLoggerType* GetAPIUsageLogger() {
  static APIUsageLoggerType func =
      IsAPIUsageDebugMode() ? &APIUsageDebug : [](const string&) {};
  return &func;
};

DDPUsageLoggerType* GetDDPUsageLogger() {
  static DDPUsageLoggerType func = [](const DDPLoggingData&) {};
  return &func;
};
} // namespace

void SetAPIUsageLogger(std::function<void(const std::string&)> logger) {
  TORCH_CHECK(logger);
  *GetAPIUsageLogger() = logger;
}

void SetPyTorchDDPUsageLogger(
    std::function<void(const DDPLoggingData&)> logger) {
  TORCH_CHECK(logger);
  *GetDDPUsageLogger() = logger;
}

void LogAPIUsage(const std::string& event) try {
  if (auto logger = GetAPIUsageLogger())
    (*logger)(event);
} catch (std::bad_function_call&) {
  // static destructor race
}

void LogPyTorchDDPUsage(const DDPLoggingData& ddpData) try {
  if (auto logger = GetDDPUsageLogger())
    (*logger)(ddpData);
} catch (std::bad_function_call&) {
  // static destructor race
}

namespace detail {
bool LogAPIUsageFakeReturn(const std::string& event) try {
  if (auto logger = GetAPIUsageLogger())
    (*logger)(event);
  return true;
} catch (std::bad_function_call&) {
  // static destructor race
  return true;
}
} // namespace detail

} // namespace c10

#if defined(C10_USE_GFLAGS) && defined(C10_USE_GLOG)
// When GLOG depends on GFLAGS, these variables are being defined in GLOG
// directly via the GFLAGS definition, so we will use DECLARE_* to declare
// them, and use them in Caffe2.
// GLOG's minloglevel
DECLARE_int32(minloglevel);
// GLOG's verbose log value.
DECLARE_int32(v);
// GLOG's logtostderr value
DECLARE_bool(logtostderr);
#endif // defined(C10_USE_GFLAGS) && defined(C10_USE_GLOG)

#if !defined(C10_USE_GLOG)
// This backward compatibility flags are in order to deal with cases where
// Caffe2 are not built with glog, but some init flags still pass in these
// flags. They may go away in the future.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_int32(minloglevel, 0, "Equivalent to glog minloglevel");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_int32(v, 0, "Equivalent to glog verbose");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_bool(logtostderr, false, "Equivalent to glog logtostderr");
#endif // !defined(c10_USE_GLOG)

#ifdef C10_USE_GLOG

// Provide easy access to the above variables, regardless whether GLOG is
// dependent on GFLAGS or not. Note that the namespace (fLI, fLB) is actually
// consistent between GLOG and GFLAGS, so we can do the below declaration
// consistently.
namespace c10 {
using fLB::FLAGS_logtostderr;
using fLI::FLAGS_minloglevel;
using fLI::FLAGS_v;
} // namespace c10

C10_DEFINE_int(
    caffe2_log_level,
    google::GLOG_WARNING,
    "The minimum log level that caffe2 will output.");

// Google glog's api does not have an external function that allows one to check
// if glog is initialized or not. It does have an internal function - so we are
// declaring it here. This is a hack but has been used by a bunch of others too
// (e.g. Torch).
namespace google {
namespace glog_internal_namespace_ {
bool IsGoogleLoggingInitialized();
} // namespace glog_internal_namespace_
} // namespace google

namespace c10 {
bool InitCaffeLogging(int* argc, char** argv) {
  if (*argc == 0)
    return true;
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
  UpdateLoggingLevelsFromFlags();
  return true;
}

void UpdateLoggingLevelsFromFlags() {
#ifdef FBCODE_CAFFE2
  // TODO(T82645998): Fix data race exposed by TSAN.
  folly::annotate_ignore_thread_sanitizer_guard g(__FILE__, __LINE__);
#endif
  // If caffe2_log_level is set and is lower than the min log level by glog,
  // we will transfer the caffe2_log_level setting to glog to override that.
  FLAGS_minloglevel = std::min(FLAGS_caffe2_log_level, FLAGS_minloglevel);
  // If caffe2_log_level is explicitly set, let's also turn on logtostderr.
  if (FLAGS_caffe2_log_level < google::GLOG_WARNING) {
    FLAGS_logtostderr = 1;
  }
  // Also, transfer the caffe2_log_level verbose setting to glog.
  if (FLAGS_caffe2_log_level < 0) {
    FLAGS_v = std::min(FLAGS_v, -FLAGS_caffe2_log_level);
  }
}

void ShowLogInfoToStderr() {
  FLAGS_logtostderr = 1;
  FLAGS_minloglevel = std::min(FLAGS_minloglevel, google::GLOG_INFO);
}
} // namespace c10

#else // !C10_USE_GLOG

#ifdef ANDROID
#include <android/log.h>
#endif // ANDROID

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_int(
    caffe2_log_level,
    c10::GLOG_WARNING,
    "The minimum log level that caffe2 will output.");

namespace c10 {
bool InitCaffeLogging(int* argc, char** argv) {
  // When doing InitCaffeLogging, we will assume that caffe's flag parser has
  // already finished.
  if (*argc == 0)
    return true;
  if (!c10::CommandLineFlagsHasBeenParsed()) {
    std::cerr << "InitCaffeLogging() has to be called after "
                 "c10::ParseCommandLineFlags. Modify your program to make sure "
                 "of this."
              << std::endl;
    return false;
  }
  if (FLAGS_caffe2_log_level > GLOG_FATAL) {
    std::cerr << "The log level of Caffe2 has to be no larger than GLOG_FATAL("
              << GLOG_FATAL << "). Capping it to GLOG_FATAL." << std::endl;
    FLAGS_caffe2_log_level = GLOG_FATAL;
  }
  return true;
}

void UpdateLoggingLevelsFromFlags() {}

void ShowLogInfoToStderr() {
  FLAGS_caffe2_log_level = GLOG_INFO;
}

MessageLogger::MessageLogger(const char* file, int line, int severity)
    : severity_(severity) {
  if (severity_ < FLAGS_caffe2_log_level) {
    // Nothing needs to be logged.
    return;
  }
#ifdef ANDROID
  tag_ = "native";
#else // !ANDROID
  tag_ = "";
#endif // ANDROID
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
          << CAFFE2_SEVERITY_PREFIX[std::min(4, GLOG_FATAL - severity_)]
          //<< (timeinfo->tm_mon + 1) * 100 + timeinfo->tm_mday
          //<< std::setfill('0')
          //<< " " << std::setw(2) << timeinfo->tm_hour
          //<< ":" << std::setw(2) << timeinfo->tm_min
          //<< ":" << std::setw(2) << timeinfo->tm_sec
          //<< "." << std::setw(9) << ns.count() % 1000000000
          << " " << c10::detail::StripBasename(std::string(file)) << ":" << line
          << "] ";
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
      ANDROID_LOG_FATAL, // LOG_FATAL
      ANDROID_LOG_ERROR, // LOG_ERROR
      ANDROID_LOG_WARN, // LOG_WARNING
      ANDROID_LOG_INFO, // LOG_INFO
      ANDROID_LOG_DEBUG, // VLOG(1)
      ANDROID_LOG_VERBOSE, // VLOG(2) .. VLOG(N)
  };
  int android_level_index = GLOG_FATAL - std::min(GLOG_FATAL, severity_);
  int level = android_log_levels[std::min(android_level_index, 5)];
  // Output the log string the Android log at the appropriate level.
  __android_log_print(level, tag_, "%s", stream_.str().c_str());
  // Indicate termination if needed.
  if (severity_ == GLOG_FATAL) {
    __android_log_print(ANDROID_LOG_FATAL, tag_, "terminating.\n");
  }
#else // !ANDROID
  if (severity_ >= FLAGS_caffe2_log_level) {
    // If not building on Android, log all output to std::cerr.
    std::cerr << stream_.str();
    // Simulating the glog default behavior: if the severity is above INFO,
    // we flush the stream so that the output appears immediately on std::cerr.
    // This is expected in some of our tests.
    if (severity_ > GLOG_INFO) {
      std::cerr << std::flush;
    }
  }
#endif // ANDROID
  if (severity_ == GLOG_FATAL) {
    DealWithFatal();
  }
}

} // namespace c10

#endif // !C10_USE_GLOG
