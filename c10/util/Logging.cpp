#include <c10/util/Backtrace.h>
#include <c10/util/Flags.h>
#include <c10/util/Lazy.h>
#include <c10/util/Logging.h>
#include <c10/util/env.h>
#ifdef FBCODE_CAFFE2
#include <folly/synchronization/SanitizeThread.h>
#endif

#ifndef _WIN32
#include <sys/time.h>
#endif

#include <algorithm>
#include <iostream>

// Common code that we use regardless of whether we use glog or not.

C10_DEFINE_bool(
    caffe2_use_fatal_for_enforce,
    false,
    "If set true, when CAFFE_ENFORCE is not met, abort instead "
    "of throwing an exception.")

namespace c10 {

namespace {
std::function<::c10::Backtrace()>& GetFetchStackTrace() {
  static std::function<::c10::Backtrace()> func = []() {
    return get_lazy_backtrace(/*frames_to_skip=*/1);
  };
  return func;
}
} // namespace

void SetStackTraceFetcher(std::function<::c10::Backtrace()> fetcher) {
  GetFetchStackTrace() = std::move(fetcher);
}

void SetStackTraceFetcher(std::function<std::string()> fetcher) {
  SetStackTraceFetcher([fetcher = std::move(fetcher)] {
    return std::make_shared<PrecomputedLazyValue<std::string>>(fetcher());
  });
}

void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller) {
  c10::Error e(file, line, condition, msg, GetFetchStackTrace()(), caller);
  if (FLAGS_caffe2_use_fatal_for_enforce) {
    LOG(FATAL) << e.msg();
  }
  throw std::move(e);
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
      file, line, condition, msg, GetFetchStackTrace()(), caller);
}

void ThrowEnforceFiniteNotMet(
    const char* file,
    const int line,
    const char* condition,
    const char* msg,
    const void* caller) {
  ThrowEnforceFiniteNotMet(file, line, condition, std::string(msg), caller);
}

namespace {

class PyTorchStyleBacktrace : public OptimisticLazyValue<std::string> {
 public:
  PyTorchStyleBacktrace(SourceLocation source_location)
      : backtrace_(GetFetchStackTrace()()), source_location_(source_location) {}

 private:
  std::string compute() const override {
    return str(
        "Exception raised from ",
        source_location_,
        " (most recent call first):\n",
        backtrace_->get());
  }

  ::c10::Backtrace backtrace_;
  SourceLocation source_location_;
};

} // namespace

// PyTorch-style error message
// (This must be defined here for access to GetFetchStackTrace)
Error::Error(SourceLocation source_location, std::string msg)
    : Error(
          std::move(msg),
          std::make_shared<PyTorchStyleBacktrace>(source_location)) {}

using APIUsageLoggerType = std::function<void(const std::string&)>;
using APIUsageMetadataLoggerType = std::function<void(
    const std::string&,
    const std::map<std::string, std::string>& metadata_map)>;
using DDPUsageLoggerType = std::function<void(const DDPLoggingData&)>;

namespace {
bool IsAPIUsageDebugMode() {
  auto val = c10::utils::get_env("PYTORCH_API_USAGE_STDERR");
  return val.has_value() && !val.value().empty(); // any non-empty value
}

void APIUsageDebug(const std::string& event) {
  // use stderr to avoid messing with glog
  std::cerr << "PYTORCH_API_USAGE " << event << '\n';
}

APIUsageLoggerType* GetAPIUsageLogger() {
  static APIUsageLoggerType func =
      IsAPIUsageDebugMode() ? &APIUsageDebug : [](const std::string&) {};
  return &func;
}

APIUsageMetadataLoggerType* GetAPIUsageMetadataLogger() {
  static APIUsageMetadataLoggerType func =
      [](const std::string&,
         const std::map<std::string, std::string>& /*metadata_map*/) {};
  return &func;
}

DDPUsageLoggerType* GetDDPUsageLogger() {
  static DDPUsageLoggerType func = [](const DDPLoggingData&) {};
  return &func;
}

auto& EventSampledHandlerRegistry() {
  static auto& registry =
      *new std::map<std::string, std::unique_ptr<EventSampledHandler>>();
  return registry;
}

} // namespace

void InitEventSampledHandlers(
    std::vector<
        std::pair<std::string_view, std::unique_ptr<EventSampledHandler>>>
        handlers) {
  static bool flag [[maybe_unused]] = [&]() {
    auto& registry = EventSampledHandlerRegistry();
    for (auto& [event, handler] : handlers) {
      auto entry = registry.find(std::string{event});
      if (entry == registry.end()) {
        entry = registry.emplace(event, nullptr).first;
      }
      entry->second = std::move(handler);
    }
    return true;
  }();
}

const std::unique_ptr<EventSampledHandler>& GetEventSampledHandler(
    std::string_view event) {
  static std::mutex guard;
  auto& registry = EventSampledHandlerRegistry();

  // The getter can be executed from different threads.
  std::lock_guard<std::mutex> lock(guard);
  auto entry = registry.find(std::string{event});
  if (entry == registry.end()) {
    entry = registry.emplace(event, nullptr).first;
  }
  return entry->second;
}

void SetAPIUsageLogger(std::function<void(const std::string&)> logger) {
  TORCH_CHECK(logger);
  *GetAPIUsageLogger() = std::move(logger);
}

void SetAPIUsageMetadataLogger(
    std::function<void(
        const std::string&,
        const std::map<std::string, std::string>& metadata_map)> logger) {
  TORCH_CHECK(logger);
  *GetAPIUsageMetadataLogger() = std::move(logger);
}

void SetPyTorchDDPUsageLogger(
    std::function<void(const DDPLoggingData&)> logger) {
  TORCH_CHECK(logger);
  *GetDDPUsageLogger() = std::move(logger);
}

static int64_t GLOBAL_RANK = -1;

void SetGlobalRank(int64_t rank) {
  GLOBAL_RANK = rank;
}

void LogAPIUsage(const std::string& event) try {
  if (auto logger = GetAPIUsageLogger())
    (*logger)(event);
  // NOLINTNEXTLINE(bugprone-empty-catch)
} catch (std::bad_function_call&) {
  // static destructor race
}

void LogAPIUsageMetadata(
    const std::string& context,
    const std::map<std::string, std::string>& metadata_map) try {
  if (auto logger = GetAPIUsageMetadataLogger())
    (*logger)(context, metadata_map);
  // NOLINTNEXTLINE(bugprone-empty-catch)
} catch (std::bad_function_call&) {
  // static destructor race
}

void LogPyTorchDDPUsage(const DDPLoggingData& ddpData) try {
  if (auto logger = GetDDPUsageLogger())
    (*logger)(ddpData);
  // NOLINTNEXTLINE(bugprone-empty-catch)
} catch (std::bad_function_call&) {
  // static destructor race
}

namespace detail {
bool LogAPIUsageFakeReturn(const std::string& event) try {
  if (auto logger = GetAPIUsageLogger())
    (*logger)(event);
  return true;
  // NOLINTNEXTLINE(bugprone-empty-catch)
} catch (std::bad_function_call&) {
  // static destructor race
  return true;
}

namespace {

void setLogLevelFlagFromEnv();

} // namespace
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
// NOLINTNEXTLINE(misc-use-internal-linkage)
C10_DEFINE_int32(minloglevel, 0, "Equivalent to glog minloglevel")
// NOLINTNEXTLINE(misc-use-internal-linkage)
C10_DEFINE_int32(v, 0, "Equivalent to glog verbose")
// NOLINTNEXTLINE(misc-use-internal-linkage)
C10_DEFINE_bool(logtostderr, false, "Equivalent to glog logtostderr")
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

MessageLogger::MessageLogger(
    const char* file,
    int line,
    int severity,
    bool exit_on_fatal)
    : stream_(), severity_(severity), exit_on_fatal_(exit_on_fatal) {}

MessageLogger::~MessageLogger() noexcept(false) {
  if (severity_ == ::google::GLOG_FATAL) {
    DealWithFatal();
  }
}

std::stringstream& MessageLogger::stream() {
  return stream_;
}

void MessageLogger::DealWithFatal() {
  if (exit_on_fatal_) {
    LOG(FATAL) << stream_.str();
  } else {
    throw c10::Error(stream_.str(), nullptr, nullptr);
  }
}

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
namespace {

void initGoogleLogging(char const* name) {
#if !defined(_MSC_VER)
  // This trick can only be used on UNIX platforms
  if (!::google::glog_internal_namespace_::IsGoogleLoggingInitialized())
#endif
  {
    ::google::InitGoogleLogging(name);
#if !defined(_MSC_VER)
    // This is never defined on Windows
    ::google::InstallFailureSignalHandler();
#endif
  }
}

} // namespace

void initLogging() {
  detail::setLogLevelFlagFromEnv();

  UpdateLoggingLevelsFromFlags();
}

bool InitCaffeLogging(int* argc, char** argv) {
  if (*argc == 0) {
    return true;
  }

  initGoogleLogging(argv[0]);

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

C10_DEFINE_int(
    caffe2_log_level,
    c10::GLOG_WARNING,
    "The minimum log level that caffe2 will output.")

namespace c10 {

void initLogging() {
  detail::setLogLevelFlagFromEnv();
}

bool InitCaffeLogging(int* argc, char** /*argv*/) {
  // When doing InitCaffeLogging, we will assume that caffe's flag parser has
  // already finished.
  if (*argc == 0)
    return true;
  if (!c10::CommandLineFlagsHasBeenParsed()) {
    std::cerr << "InitCaffeLogging() has to be called after "
                 "c10::ParseCommandLineFlags. Modify your program to make sure "
                 "of this."
              << '\n';
    return false;
  }
  if (FLAGS_caffe2_log_level > GLOG_FATAL) {
    std::cerr << "The log level of Caffe2 has to be no larger than GLOG_FATAL("
              << GLOG_FATAL << "). Capping it to GLOG_FATAL." << '\n';
    FLAGS_caffe2_log_level = GLOG_FATAL;
  }
  return true;
}

void UpdateLoggingLevelsFromFlags() {}

void ShowLogInfoToStderr() {
  FLAGS_caffe2_log_level = GLOG_INFO;
}

MessageLogger::MessageLogger(
    const char* file,
    int line,
    int severity,
    bool exit_on_fatal)
    : severity_(severity), exit_on_fatal_(exit_on_fatal) {
  if (severity_ < FLAGS_caffe2_log_level) {
    // Nothing needs to be logged.
    return;
  }

  time_t rawtime = 0;
  time(&rawtime);

#ifndef _WIN32
  struct tm raw_timeinfo = {0};
  struct tm* timeinfo = &raw_timeinfo;
  localtime_r(&rawtime, timeinfo);
#else
  // is thread safe on Windows
  struct tm* timeinfo = localtime(&rawtime);
#endif

#ifndef _WIN32
  // Get the current nanoseconds since epoch
  struct timespec ts = {0};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  long ns = ts.tv_nsec;
#else
  long ns = 0;
#endif

  if (GLOBAL_RANK != -1) {
    stream_ << "[rank" << GLOBAL_RANK << "]:";
  }
  stream_ << "[" << CAFFE2_SEVERITY_PREFIX[std::min(4, GLOG_FATAL - severity_)]
          << (timeinfo->tm_mon + 1) * 100 + timeinfo->tm_mday
          << std::setfill('0') << " " << std::setw(2) << timeinfo->tm_hour
          << ":" << std::setw(2) << timeinfo->tm_min << ":" << std::setw(2)
          << timeinfo->tm_sec << "." << std::setw(9) << ns << " "
          << c10::detail::StripBasename(std::string(file)) << ":" << line
          << "] ";
}

// Output the contents of the stream to the proper channel on destruction.
MessageLogger::~MessageLogger() noexcept(false) {
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

std::stringstream& MessageLogger::stream() {
  return stream_;
}

void MessageLogger::DealWithFatal() {
  if (exit_on_fatal_) {
    abort();
  } else {
    throw c10::Error(stream_.str(), nullptr, nullptr);
  }
}

} // namespace c10

#endif // !C10_USE_GLOG

namespace c10::detail {
namespace {

void setLogLevelFlagFromEnv() {
  auto level_env = c10::utils::get_env("TORCH_CPP_LOG_LEVEL");

  // Not set, fallback to the default level (i.e. WARNING).
  std::string level{level_env.has_value() ? level_env.value() : ""};
  if (level.empty()) {
    return;
  }

  std::transform(
      level.begin(), level.end(), level.begin(), [](unsigned char c) {
        return toupper(c);
      });

  if (level == "0" || level == "INFO") {
    FLAGS_caffe2_log_level = 0;

    return;
  }
  if (level == "1" || level == "WARNING") {
    FLAGS_caffe2_log_level = 1;

    return;
  }
  if (level == "2" || level == "ERROR") {
    FLAGS_caffe2_log_level = 2;

    return;
  }
  if (level == "3" || level == "FATAL") {
    FLAGS_caffe2_log_level = 3;

    return;
  }

  std::cerr
      << "`TORCH_CPP_LOG_LEVEL` environment variable cannot be parsed. Valid values are "
         "`INFO`, `WARNING`, `ERROR`, and `FATAL` or their numerical equivalents `0`, `1`, "
         "`2`, and `3`."
      << '\n';
}

} // namespace
} // namespace c10::detail
