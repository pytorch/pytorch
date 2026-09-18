/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Config.h"
#include "ThrowUtil.h"

#include <cstdlib>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <chrono>
#include <ctime>
#include <functional>
#include <mutex>
#include <ostream>
#include <string_view>
#include <utility>

#include "Logger.h"
#include "ThreadUtil.h"

using namespace std::chrono;

using std::string;
using std::vector;

namespace KINETO_NAMESPACE {

#if __cplusplus < 201703L
constexpr std::chrono::milliseconds Config::kControllerIntervalMsecs;
#endif

constexpr milliseconds kDefaultActivitiesProfileDurationMSecs(500);
constexpr int64_t kDefaultActivitiesMaxGpuBufferSize(128 * 1024 * 1024);
constexpr seconds kDefaultActivitiesWarmupDurationSecs(5);
constexpr seconds kMaxRequestAge(10);
constexpr seconds kDefaultOnDemandConfigUpdateIntervalSecs(5);
// 3200000 is the default value set by CUPTI
constexpr size_t kDefaultCuptiDeviceBufferSize(3200000);
// Default value set by CUPTI is 250
constexpr size_t kDefaultCuptiDeviceBufferPoolLimit(20);

// Activity Profiler
constexpr char kActivitiesEnabledKey[] = "ACTIVITIES_ENABLED";
constexpr char kCuptiPerThreadBufferEnabledKey[] =
    "CUPTI_PER_THREAD_BUFFER_ENABLED";
constexpr char kPerformanceMetricsKey[] = "PERFORMANCE_METRICS";
constexpr char kPerformanceMetricsDeviceIdKey[] =
    "PERFORMANCE_METRICS_DEVICE_ID";
constexpr char kActivityTypesKey[] = "ACTIVITY_TYPES";
constexpr char kActivitiesLogFileKey[] = "ACTIVITIES_LOG_FILE";
constexpr char kActivitiesDurationKey[] = "ACTIVITIES_DURATION_SECS";
constexpr char kActivitiesDurationMsecsKey[] = "ACTIVITIES_DURATION_MSECS";
constexpr char kActivitiesWarmupDurationSecsKey[] =
    "ACTIVITIES_WARMUP_PERIOD_SECS";
constexpr char kActivitiesMaxGpuBufferSizeKey[] =
    "ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB";
constexpr char kActivitiesDisplayCudaSyncWaitEvents[] =
    "ACTIVITIES_DISPLAY_CUDA_SYNC_WAIT_EVENTS";

// Client Interface
// TODO: keep supporting these older config options, deprecate in the future
// using replacements.
constexpr char kClientInterfaceEnableOpInputsCollection[] =
    "CLIENT_INTERFACE_ENABLE_OP_INPUTS_COLLECTION";
constexpr char kPythonStackTrace[] = "PYTHON_STACK_TRACE";
// Profiler Config Options
constexpr char kProfileReportInputShapes[] = "PROFILE_REPORT_INPUT_SHAPES";
constexpr char kProfileProfileMemory[] = "PROFILE_PROFILE_MEMORY";
constexpr char kProfileWithStack[] = "PROFILE_WITH_STACK";
constexpr char kProfileWithFlops[] = "PROFILE_WITH_FLOPS";
constexpr char kProfileWithModules[] = "PROFILE_WITH_MODULES";

constexpr char kActivitiesWarmupIterationsKey[] =
    "ACTIVITIES_WARMUP_ITERATIONS";
constexpr char kActivitiesIterationsKey[] = "ACTIVITIES_ITERATIONS";

// Memory Profiler
constexpr char kProfileMemory[] = "PROFILE_MEMORY";
constexpr char kProfileMemoryDuration[] = "PROFILE_MEMORY_DURATION_MSECS";

// Roctracer
constexpr char kRoctracerSetMaxEvents[] = "ROCTRACER_MAX_EVENTS";

// Common

// Client-side timestamp used for synchronized start across hosts for
// distributed workloads.
// Specified in milliseconds Unix time (milliseconds since epoch).
// To use, compute a future timestamp as follows:
//    * C++: <delay_ms> + duration_cast<milliseconds>(
//               system_clock::now().time_since_epoch()).count()
//    * Python: <delay_ms> + int(time.time() * 1000)
//    * Bash: $((<delay_ms> + $(date +%s%3N)))
//    * Bash: $(date -d "$time + <delay_secs>seconds" +%s%3N)
// If used for a tracing request, timestamp must be far enough in the future
// to accommodate ACTIVITIES_WARMUP_PERIOD_SECS as well as any delays in
// propagating the request to the profiler.
// If the request can not be honored, it is up to the profilers to report
// an error somehow - no checks are done at config parse time.
// Note PROFILE_START_ITERATION has higher precedence
constexpr char kProfileStartTimeKey[] = "PROFILE_START_TIME";
// Alternatively if the application supports reporting iterations
// start the profile at specific iteration. If the iteration count
// is >= this value the profile is started immediately.
// A value >= 0 is valid for this config option to take effect.
// Note PROFILE_START_ITERATION will take precedence over PROFILE_START_TIME.
constexpr char kProfileStartIterationKey[] = "PROFILE_START_ITERATION";

// Users can also start the profile on an integer multiple of the config
// value PROFILE_START_ITERATION_ROUNDUP. This knob behaves similar to
// PROFILE_START_ITERATION but instead of saying : "start collection trace on
// iteration 500", one can configure it to "start collecting trace on the next
// 100th iteration".
//
// For example,
//   PROFILE_START_ITERATION_ROUNDUP = 1000, and the current iteration is 2010
//   The profile will then be collected on the next multiple of 1000 ie. 3000
// Note PROFILE_START_ITERATION_ROUNDUP will also take precedence over
// PROFILE_START_TIME.
constexpr char kProfileStartIterationRoundUpKey[] =
    "PROFILE_START_ITERATION_ROUNDUP";

constexpr char kRequestTraceID[] = "REQUEST_TRACE_ID";
constexpr char kRequestGroupTraceID[] = "REQUEST_GROUP_TRACE_ID";

// Enable communication through IPC Fabric
// and disable thrift communication with dynolog daemon
constexpr char kEnableIpcFabricKey[] = "ENABLE_IPC_FABRIC";
// Period to pull on-demand config from dynolog daemon
constexpr char kOnDemandConfigUpdateIntervalSecsKey[] =
    "ON_DEMAND_CONFIG_UPDATE_INTERVAL_SECS";

// Verbose log level
// The actual glog is not used and --v and --vmodule has no effect.
// Instead set the verbose level and modules in the config file.
constexpr char kLogVerboseLevelKey[] = "VERBOSE_LOG_LEVEL";
// By default, all modules will log verbose messages >= verboseLogLevel.
// But to reduce noise we can specify one or more modules of interest.
// A module is a C/C++ object file (source file name),
// Example argument: ActivityProfiler.cpp,output_json.cpp
constexpr char kLogVerboseModulesKey[] = "VERBOSE_LOG_MODULES";

constexpr char kCustomConfigKey[] = "CUSTOM_CONFIG";

namespace {

struct FactoryMap {
  void addFactory(
      std::string name,
      std::function<AbstractConfig*(Config&)> factory) {
    std::scoped_lock lock(lock_);
    factories_.emplace(std::move(name), std::move(factory));
  }

  void addFeatureConfigs(Config& cfg) {
    std::scoped_lock lock(lock_);
    for (const auto& p : factories_) {
      cfg.addFeature(p.first, p.second(cfg));
    }
  }

  // Config factories are shared between objects and since
  // config objects can be created by multiple threads, we need a lock.
  std::mutex lock_;
  std::map<std::string, std::function<AbstractConfig*(Config&)>> factories_;
};

std::shared_ptr<FactoryMap> configFactories() {
  // Ensure this is safe to call during shutdown, even as static
  // destructors are invoked. getStaticObjectLifetimeHandle hangs onto
  // FactoryMap delaying its destruction.
  static auto factories = std::make_shared<FactoryMap>();
  static std::weak_ptr<FactoryMap> weak_ptr = factories;
  return weak_ptr.lock();
}

} // namespace

void Config::addConfigFactory(
    std::string name,
    std::function<AbstractConfig*(Config&)> factory) {
  auto factories = configFactories();
  if (factories) {
    factories->addFactory(std::move(name), std::move(factory));
  }
}

static string defaultTraceFileName() {
  return fmt::format("/tmp/libkineto_activities_{}.json", processId());
}

static string defaultMemoryTraceFileName() {
  return fmt::format("/tmp/memory_snapshot_{}.pickle", processId());
}

namespace {

// Dir that on-demand trace files are restricted to. Defaults to
// /tmp; overridable locally via KINETO_ONDEMAND_TRACE_DIR.
constexpr std::string_view kDefaultOnDemandTraceDir = "/tmp/";

const string& allowedOnDemandTraceDir() {
  static const string kDir = [] {
    const char* env = std::getenv("KINETO_ONDEMAND_TRACE_DIR");
    string d = (env != nullptr && *env != '\0')
        ? string(env)
        : string(kDefaultOnDemandTraceDir);
    if (d.back() != '/') {
      d.push_back('/');
    }
    return d;
  }();
  return kDir;
}

// Allowed only if under the allowed dir and free of ".." traversal.
bool isAllowedOnDemandTraceFile(const string& path) {
  const string& dir = allowedOnDemandTraceDir();
  return path.starts_with(dir) && path.find("..") == string::npos;
}

} // namespace

Config::Config()
    : verboseLogLevel_(-1),
      activityProfilerEnabled_(true),
      perThreadBufferEnabled_(false),
      activitiesLogFile_(defaultTraceFileName()),
      activitiesLogUrl_(fmt::format("file://{}", activitiesLogFile_)),
      activitiesMaxGpuBufferSize_(kDefaultActivitiesMaxGpuBufferSize),
      activitiesWarmupDuration_(kDefaultActivitiesWarmupDurationSecs),
      activitiesWarmupIterations_(0),
      activitiesCudaSyncWaitEvents_(true),
      activitiesDuration_(kDefaultActivitiesProfileDurationMSecs),
      activitiesRunIterations_(0),
      activitiesOnDemandTimestamp_(milliseconds(0)),
      profileStartTime_(milliseconds(0)),
      profileStartIteration_(-1),
      profileStartIterationRoundUp_(-1),
      enableIpcFabric_(false),
      onDemandConfigUpdateIntervalSecs_(
          kDefaultOnDemandConfigUpdateIntervalSecs),
      cuptiDeviceBufferSize_(kDefaultCuptiDeviceBufferSize),
      cuptiDeviceBufferPoolLimit_(kDefaultCuptiDeviceBufferPoolLimit) {
  auto factories = configFactories();
  if (factories) {
    factories->addFeatureConfigs(*this);
  }
#if __linux__
  enableIpcFabric_ = libkineto::isDaemonEnvVarSet();
#endif
}

#if __linux__
bool isDaemonEnvVarSet() {
  static bool rc = [] {
    void* ptr = getenv(kUseDaemonEnvVar);
    return ptr != nullptr;
  }();
  return rc;
}
#else
bool isDaemonEnvVarSet() {
  return false;
}
#endif

std::shared_ptr<void> Config::getStaticObjectsLifetimeHandle() {
  return configFactories();
}

seconds Config::maxRequestAge() const {
  return kMaxRequestAge;
}

static std::string getTimeStr(time_point<system_clock> t) {
  std::time_t t_c = system_clock::to_time_t(t);
  std::tm tm{};
  get_local_time(&t_c, &tm);
  return fmt::format("{:%H:%M:%S}", tm);
}

static time_point<system_clock> handleProfileStartTime(int64_t start_time_ms) {
  // If 0, return 0, so that AbstractConfig::parse can fix the timestamp later.
  if (start_time_ms == 0) {
    return time_point<system_clock>(milliseconds(0));
  }

  auto t = time_point<system_clock>(milliseconds(start_time_ms));
  // This should check that ProfileStartTime is in the future with
  // enough time for warm-up.
  // Unfortunately, warm-up duration is unknown at this point.
  // But we can still check that the start time is not in the past.
  auto now = system_clock::now();
  if ((now - t) > kMaxRequestAge) {
    KINETO_THROW(
        std::invalid_argument,
        fmt::format(
            "Invalid {}: {} - start time is more than {}s in the past",
            kProfileStartTimeKey,
            getTimeStr(t),
            kMaxRequestAge.count()));
  }
  return t;
}

void Config::setActivityTypes(
    const std::vector<std::string>& selected_activities) {
  selectedActivityTypes_.clear();
  if (!selected_activities.empty()) {
    for (const auto& activity : selected_activities) {
      if (activity.empty()) {
        continue;
      }
      selectedActivityTypes_.insert(toActivityType(activity));
    }
  }
}

bool Config::handleOption(const std::string& name, std::string& val) {
  if (!name.compare(kActivitiesDurationKey)) {
    activitiesDuration_ = duration_cast<milliseconds>(seconds(toInt32(val)));
    activitiesOnDemandTimestamp_ = timestamp();
  } else if (!name.compare(kActivityTypesKey)) {
    vector<string> activity_types = splitAndTrim(toLower(val), ',');
    setActivityTypes(activity_types);
  } else if (!name.compare(kActivitiesDurationMsecsKey)) {
    activitiesDuration_ = milliseconds(toInt32(val));
    activitiesOnDemandTimestamp_ = timestamp();
  } else if (!name.compare(kActivitiesIterationsKey)) {
    activitiesRunIterations_ = toInt32(val);
    activitiesOnDemandTimestamp_ = timestamp();
  } else if (!name.compare(kLogVerboseLevelKey)) {
    verboseLogLevel_ = toInt32(val);
  } else if (!name.compare(kLogVerboseModulesKey)) {
    verboseLogModules_ = splitAndTrim(val, ',');
  } else if (!name.compare(kActivitiesEnabledKey)) {
    activityProfilerEnabled_ = toBool(val);
  } else if (!name.compare(kCuptiPerThreadBufferEnabledKey)) {
    perThreadBufferEnabled_ = toBool(val);
  } else if (!name.compare(kPerformanceMetricsKey)) {
    performanceMetricNames_ = splitAndTrim(val, ',');
  } else if (!name.compare(kPerformanceMetricsDeviceIdKey)) {
    performanceMetricsDeviceId_ = toInt32(val);
  } else if (!name.compare(kProfileMemory)) {
    memoryProfilerEnabled_ = toBool(val);
    if (memoryProfilerEnabled_) {
      activitiesLogFile_ = defaultMemoryTraceFileName();
    }
  } else if (!name.compare(kProfileMemoryDuration)) {
    profileMemoryDuration_ = toInt32(val);
  } else if (!name.compare(kActivitiesLogFileKey)) {
    if (onDemand_ && !isAllowedOnDemandTraceFile(val)) {
      LOG(WARNING) << "Ignoring on-demand " << kActivitiesLogFileKey
                   << " outside allowed directory " << allowedOnDemandTraceDir()
                   << ": " << val << " (trace will use the default path)";
    } else {
      activitiesLogFile_ = val;
      activitiesLogUrl_ = fmt::format("file://{}", val);
      size_t jidx = activitiesLogUrl_.find(".pt.trace.json");
      if (jidx != std::string::npos) {
        activitiesLogUrl_.replace(
            jidx, 14, fmt::format("_{}.pt.trace.json", processId()));
      } else {
        jidx = activitiesLogUrl_.find(".json");
        if (jidx != std::string::npos) {
          activitiesLogUrl_.replace(
              jidx, 5, fmt::format("_{}.json", processId()));
        }
      }
    }
    activitiesOnDemandTimestamp_ = timestamp();
  } else if (!name.compare(kActivitiesMaxGpuBufferSizeKey)) {
    activitiesMaxGpuBufferSize_ =
        static_cast<int64_t>(toInt32(val)) * 1024 * 1024;
  } else if (!name.compare(kActivitiesWarmupDurationSecsKey)) {
    activitiesWarmupDuration_ = seconds(toInt32(val));
  } else if (!name.compare(kActivitiesWarmupIterationsKey)) {
    activitiesWarmupIterations_ = toInt32(val);
  } else if (!name.compare(kActivitiesDisplayCudaSyncWaitEvents)) {
    activitiesCudaSyncWaitEvents_ = toBool(val);
  } else if (!name.compare(kRequestTraceID)) {
    requestTraceID_ = val;
  } else if (!name.compare(kRequestGroupTraceID)) {
    requestGroupTraceID_ = val;
  } else if (!name.compare(kRoctracerSetMaxEvents)) {
    maxEvents_ = toInt32(val);
  }

  // TODO: Deprecate Client Interface
  else if (!name.compare(kClientInterfaceEnableOpInputsCollection)) {
    enableReportInputShapes_ = toBool(val);
  } else if (!name.compare(kPythonStackTrace)) {
    enableWithStack_ = toBool(val);
  }

  // Profiler Config
  else if (!name.compare(kProfileReportInputShapes)) {
    enableReportInputShapes_ = toBool(val);
  } else if (!name.compare(kProfileProfileMemory)) {
    enableProfileMemory_ = toBool(val);
  } else if (!name.compare(kProfileWithStack)) {
    enableWithStack_ = toBool(val);
  } else if (!name.compare(kProfileWithFlops)) {
    enableWithFlops_ = toBool(val);
  } else if (!name.compare(kProfileWithModules)) {
    enableWithModules_ = toBool(val);
  }

  // Common
  else if (!name.compare(kProfileStartTimeKey)) {
    profileStartTime_ = handleProfileStartTime(toInt64(val));
  } else if (!name.compare(kProfileStartIterationKey)) {
    profileStartIteration_ = toInt32(val);
  } else if (!name.compare(kProfileStartIterationRoundUpKey)) {
    profileStartIterationRoundUp_ = toInt32(val);
  } else if (!name.compare(kEnableIpcFabricKey)) {
    enableIpcFabric_ = toBool(val);
  } else if (!name.compare(kOnDemandConfigUpdateIntervalSecsKey)) {
    onDemandConfigUpdateIntervalSecs_ = seconds(toInt32(val));
  } else if (!name.compare(kCustomConfigKey)) {
    customConfig_ = val;
  } else {
    return false;
  }
  return true;
}

void Config::updateActivityProfilerRequestReceivedTime() {
  activitiesOnDemandTimestamp_ = system_clock::now();
}

void Config::setClientDefaults() {
  AbstractConfig::setClientDefaults();
  activitiesLogToMemory_ = true;
}

void Config::validate(
    const time_point<system_clock>& fallbackProfileStartTime) {
  if (!hasProfileStartTime()) {
    VLOG(0)
        << "No explicit timestamp has been set. "
        << "Defaulting it to now + activitiesWarmupDuration with a buffer of double the period of the monitoring thread.";
    profileStartTime_ = fallbackProfileStartTime + activitiesWarmupDuration() +
        2 * Config::kControllerIntervalMsecs;
  }

  if (profileStartIterationRoundUp_ == 0) {
    // setting to 0 will mess up modulo arithmetic, set it to -1 so it has no
    // effect
    LOG(WARNING) << "Profiler start iteration round up should be >= 1.";
    profileStartIterationRoundUp_ = -1;
  }

  if (profileStartIterationRoundUp_ > 0 && !hasProfileStartIteration()) {
    VLOG(0) << "Setting profiler start iteration to 0 so this config is "
            << "triggered via iteration count.";
    profileStartIteration_ = 0;
  }

  if (selectedActivityTypes_.empty()) {
    selectDefaultActivityTypes();
  }
  setActivityDependentConfig();
}

void Config::printActivityProfilerConfig(std::ostream& s) const {
  fmt::print(s, "  Log file: {}\n", activitiesLogFile());
  if (hasProfileStartIteration()) {
    fmt::print(
        s,
        "  Trace start Iteration: {}\n"
        "  Trace warmup Iterations: {}\n"
        "  Trace profile Iterations: {}\n",
        profileStartIteration(),
        activitiesWarmupIterations(),
        activitiesRunIterations());
    if (profileStartIterationRoundUp() > 0) {
      fmt::print(
          "  Trace start iteration roundup : {}\n",
          profileStartIterationRoundUp());
    }
  } else if (hasProfileStartTime()) {
    std::time_t t_c = system_clock::to_time_t(requestTimestamp());
    std::tm tm{};
    get_local_time(&t_c, &tm);
    fmt::print(
        s,
        "  Trace start time: {:%Y-%m-%d %H:%M:%S}\n"
        "  Trace duration: {}ms\n"
        "  Warmup duration: {}s\n",
        tm,
        activitiesDuration().count(),
        activitiesWarmupDuration().count());
  }

  fmt::print(
      s,
      "  Max GPU buffer size: {:.0f}MB\n",
      static_cast<double>(activitiesMaxGpuBufferSize()) / 1024.0 / 1024.0);

  std::vector<std::string> activities;
  activities.reserve(selectedActivityTypes_.size());
  for (const auto& activity : selectedActivityTypes_) {
    activities.emplace_back(toString(activity));
  }
  fmt::print(s, "  Enabled activities: {}\n", fmt::join(activities, ","));

  AbstractConfig::printActivityProfilerConfig(s);
}

void Config::setActivityDependentConfig() {
  AbstractConfig::setActivityDependentConfig();
}

// Returns a reference to the perfetto trace enabled flag.
// Default is false. Downstream consumers override at startup.
bool& get_perfetto_trace_enabled() {
  static bool perfetto_trace_enabled = false;
  return perfetto_trace_enabled;
}

// Returns a reference to the perfetto packet compression enabled flag.
// Default is true (compression is on). Downstream consumers override at
// startup.
bool& get_perfetto_packet_compression_enabled() {
  static bool perfetto_packet_compression_enabled = true;
  return perfetto_packet_compression_enabled;
}

} // namespace KINETO_NAMESPACE
