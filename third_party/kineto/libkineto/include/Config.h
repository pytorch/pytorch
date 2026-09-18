/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "AbstractConfig.h"
#include "ActivityType.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace libkineto {

class Config : public AbstractConfig {
 public:
  Config();
  Config& operator=(const Config&) = delete;
  Config(Config&&) = delete;
  Config& operator=(Config&&) = delete;
  ~Config() override = default;

  // Return a full copy including feature config object
  [[nodiscard]] std::unique_ptr<Config> clone() const {
    auto cfg = std::unique_ptr<Config>(new Config(*this));
    cloneFeaturesInto(*cfg);
    return cfg;
  }

  bool handleOption(const std::string& name, std::string& val) override;

  void setClientDefaults() override;

  [[nodiscard]] bool activityProfilerEnabled() const {
    return activityProfilerEnabled_ ||
        activitiesOnDemandTimestamp_.time_since_epoch().count() > 0;
  }

  // Log activitiy trace to this file
  [[nodiscard]] const std::string& activitiesLogFile() const {
    return activitiesLogFile_;
  }

  // Log activitiy trace to this url
  [[nodiscard]] const std::string& activitiesLogUrl() const {
    return activitiesLogUrl_;
  }

  void setActivitiesLogUrl(const std::string& url) {
    activitiesLogUrl_ = url;
  }

  // Called for configs from the daemon IPC path. See onDemand_.
  void setOnDemand(bool onDemand) {
    onDemand_ = onDemand;
  }

  [[nodiscard]] bool activitiesLogToMemory() const {
    return activitiesLogToMemory_;
  }

  // The types of activities selected in the configuration file
  [[nodiscard]] const std::set<ActivityType>& selectedActivityTypes() const {
    return selectedActivityTypes_;
  }

  // Set the types of activities to be traced
  [[nodiscard]] bool perThreadBufferEnabled() const {
    return perThreadBufferEnabled_;
  }

  void setSelectedActivityTypes(const std::set<ActivityType>& types) {
    selectedActivityTypes_ = types;
  }

  [[nodiscard]] bool isReportInputShapesEnabled() const {
    return enableReportInputShapes_;
  }

  [[nodiscard]] bool isProfileMemoryEnabled() const {
    return enableProfileMemory_;
  }

  [[nodiscard]] bool isWithStackEnabled() const {
    return enableWithStack_;
  }

  [[nodiscard]] bool isWithFlopsEnabled() const {
    return enableWithFlops_;
  }

  [[nodiscard]] bool isWithModulesEnabled() const {
    return enableWithModules_;
  }

  // Trace for this long
  [[nodiscard]] std::chrono::milliseconds activitiesDuration() const {
    return activitiesDuration_;
  }

  // Trace for this many iterations, determined by external API
  [[nodiscard]] int activitiesRunIterations() const {
    return activitiesRunIterations_;
  }

  [[nodiscard]] int64_t activitiesMaxGpuBufferSize() const {
    return activitiesMaxGpuBufferSize_;
  }

  [[nodiscard]] std::chrono::seconds activitiesWarmupDuration() const {
    return activitiesWarmupDuration_;
  }

  [[nodiscard]] int activitiesWarmupIterations() const {
    return activitiesWarmupIterations_;
  }

  // Show CUDA Synchronization Stream Wait Events
  [[nodiscard]] bool activitiesCudaSyncWaitEvents() const {
    return activitiesCudaSyncWaitEvents_;
  }

  void setActivitiesCudaSyncWaitEvents(bool enable) {
    activitiesCudaSyncWaitEvents_ = enable;
  }

  [[nodiscard]] std::chrono::time_point<std::chrono::system_clock>
  requestTimestamp() const {
    return profileStartTime_;
  }

  [[nodiscard]] bool hasProfileStartTime() const {
    return profileStartTime_.time_since_epoch().count() > 0;
  }

  [[nodiscard]] int profileStartIteration() const {
    return profileStartIteration_;
  }

  [[nodiscard]] bool hasProfileStartIteration() const {
    return profileStartIteration_ >= 0 && activitiesRunIterations_ > 0;
  }

  void setProfileStartIteration(int iter) {
    profileStartIteration_ = iter;
  }

  [[nodiscard]] int profileStartIterationRoundUp() const {
    return profileStartIterationRoundUp_;
  }

  // calculate the start iteration accounting for warmup
  [[nodiscard]] int startIterationIncludingWarmup() const {
    if (!hasProfileStartIteration()) {
      return -1;
    }
    return profileStartIteration_ - activitiesWarmupIterations_;
  }

  [[nodiscard]] std::chrono::seconds maxRequestAge() const;

  // All VLOG* macros will log if the verbose log level is >=
  // the verbosity specified for the verbose log message.
  // Default value is -1, so messages with log level 0 will log by default.
  [[nodiscard]] int verboseLogLevel() const {
    return verboseLogLevel_;
  }

  // Modules for which verbose logging is enabled.
  // If empty, logging is enabled for all modules.
  [[nodiscard]] const std::vector<std::string>& verboseLogModules() const {
    return verboseLogModules_;
  }

  [[nodiscard]] bool ipcFabricEnabled() const {
    return enableIpcFabric_;
  }

  [[nodiscard]] std::chrono::seconds onDemandConfigUpdateIntervalSecs() const {
    return onDemandConfigUpdateIntervalSecs_;
  }

  static std::chrono::milliseconds alignUp(
      std::chrono::milliseconds duration,
      std::chrono::milliseconds alignment) {
    duration += alignment;
    return duration - (duration % alignment);
  }

  [[nodiscard]] std::chrono::time_point<std::chrono::system_clock>
  activityProfilerRequestReceivedTime() const {
    return activitiesOnDemandTimestamp_;
  }

  static constexpr std::chrono::milliseconds kControllerIntervalMsecs{1000};

  // Users may request and set trace id and group trace id.
  [[nodiscard]] const std::string& requestTraceID() const {
    return requestTraceID_;
  }

  void setRequestTraceID(const std::string& tid) {
    requestTraceID_ = tid;
  }

  [[nodiscard]] const std::string& requestGroupTraceID() const {
    return requestGroupTraceID_;
  }

  void setRequestGroupTraceID(const std::string& gtid) {
    requestGroupTraceID_ = gtid;
  }

  [[nodiscard]] size_t cuptiDeviceBufferSize() const {
    return cuptiDeviceBufferSize_;
  }

  [[nodiscard]] size_t cuptiDeviceBufferPoolLimit() const {
    return cuptiDeviceBufferPoolLimit_;
  }

  [[nodiscard]] const std::vector<std::string>& performanceMetricNames() const {
    return performanceMetricNames_;
  }

  [[nodiscard]] int32_t performanceMetricsDeviceId() const {
    return performanceMetricsDeviceId_;
  }

  [[nodiscard]] bool memoryProfilerEnabled() const {
    return memoryProfilerEnabled_;
  }

  [[nodiscard]] int profileMemoryDuration() const {
    return profileMemoryDuration_;
  }
  void updateActivityProfilerRequestReceivedTime();

  void printActivityProfilerConfig(std::ostream& s) const override;
  void setActivityDependentConfig() override;

  void validate(const std::chrono::time_point<std::chrono::system_clock>&
                    fallbackProfileStartTime) override;

  static void addConfigFactory(
      std::string name,
      std::function<AbstractConfig*(Config&)> factory);

  void print(std::ostream& s) const;

  // Config relies on some state with global static lifetime. If other
  // threads are using the config, it's possible that the global state
  // is destroyed before the threads stop. By hanging onto this handle,
  // correct destruction order can be ensured.
  static std::shared_ptr<void> getStaticObjectsLifetimeHandle();

  [[nodiscard]] bool getTSCTimestampFlag() const {
    return useTSCTimestamp_;
  }

  void setTSCTimestampFlag(bool flag) {
    useTSCTimestamp_ = flag;
  }

  [[nodiscard]] const std::string& getCustomConfig() const {
    return customConfig_;
  }

  [[nodiscard]] uint32_t maxEvents() const {
    return maxEvents_;
  }

 private:
  explicit Config(const Config& other) = default;

  AbstractConfig* cloneDerived(
      [[maybe_unused]] AbstractConfig& parent) const override {
    // Clone from AbstractConfig not supported
    assert(false);
    return nullptr;
  }

  // Adds valid activity types from the user defined string list in the
  // configuration file
  void setActivityTypes(const std::vector<std::string>& selected_activities);

  // Sets the default activity types to be traced
  void selectDefaultActivityTypes() {
    // If the user has not specified an activity list, add all types
    for (ActivityType t : defaultActivityTypes()) {
      selectedActivityTypes_.insert(t);
    }
  }

  int verboseLogLevel_;
  std::vector<std::string> verboseLogModules_;

  // Activity profiler
  bool activityProfilerEnabled_;

  // Enable per-thread buffer
  bool perThreadBufferEnabled_;
  std::set<ActivityType> selectedActivityTypes_;

  // The activity profiler settings are all on-demand
  std::string activitiesLogFile_;

  std::string activitiesLogUrl_;

  // Log activities to memory buffer
  bool activitiesLogToMemory_{false};

  // Restricts trace output path when set (untrusted on-demand config).
  bool onDemand_{false};

  int64_t activitiesMaxGpuBufferSize_;
  std::chrono::seconds activitiesWarmupDuration_;
  int activitiesWarmupIterations_;
  bool activitiesCudaSyncWaitEvents_;

  // Enable Profiler Config Options
  // Temporarily disable shape collection until we re-roll out the feature for
  // on-demand cases
  bool enableReportInputShapes_{false};
  bool enableProfileMemory_{false};
  bool enableWithStack_{false};
  bool enableWithFlops_{false};
  bool enableWithModules_{false};

  // Profile for specified iterations and duration
  std::chrono::milliseconds activitiesDuration_;
  int activitiesRunIterations_;

  // Below are not used
  // Use this net name for iteration count
  std::string activitiesExternalAPIIterationsTarget_;
  // Only profile nets that includes this in the name
  std::vector<std::string> activitiesExternalAPIFilter_;
  // Only profile nets with at least this many operators
  int activitiesExternalAPINetSizeThreshold_;
  // Only profile nets with at least this many GPU operators
  int activitiesExternalAPIGpuOpCountThreshold_;
  // Last activity profiler request
  std::chrono::time_point<std::chrono::system_clock>
      activitiesOnDemandTimestamp_;

  // ActivityProfilers are triggered by either:
  // Synchronized start timestamps
  std::chrono::time_point<std::chrono::system_clock> profileStartTime_;
  // Or start iterations.
  int profileStartIteration_;
  int profileStartIterationRoundUp_;

  // Enable IPC Fabric instead of thrift communication
  bool enableIpcFabric_;
  std::chrono::seconds onDemandConfigUpdateIntervalSecs_;

  // Logger Metadata
  std::string requestTraceID_;
  std::string requestGroupTraceID_;

  // CUPTI Device Buffer
  size_t cuptiDeviceBufferSize_;
  size_t cuptiDeviceBufferPoolLimit_;

  // Performance metrics
  std::vector<std::string> performanceMetricNames_;
  int32_t performanceMetricsDeviceId_{-1};

  // CUPTI Timestamp Format
  bool useTSCTimestamp_{true};

  // Memory Profiler
  bool memoryProfilerEnabled_{false};
  int profileMemoryDuration_{1000};

  // Used to flexibly configure some custom options, especially for custom
  // backends. How to parse this string is handled by the custom backend.
  std::string customConfig_;

  // Roctracer settings
  uint32_t maxEvents_{5000000};
};

constexpr char kUseDaemonEnvVar[] = "KINETO_USE_DAEMON";

bool isDaemonEnvVarSet();

// Returns a reference to the Perfetto trace enabled flag.
// When true, a consumer writes a Perfetto-native trace via the
// Perfetto SDK alongside other output formats.
bool& get_perfetto_trace_enabled();

// Returns a reference to the Perfetto packet compression enabled flag.
// When true, in-process Perfetto SDK tracing sessions configure
// `TraceConfig.compression_type = COMPRESSION_TYPE_DEFLATE`, producing
// a smaller .pftrace at the cost of some CPU. The Perfetto UI and
// trace_processor decompress transparently.
bool& get_perfetto_packet_compression_enabled();

} // namespace libkineto
