/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude

#include "GenericTraceActivity.h"
#include "IActivityProfiler.h"
#include "ThreadUtil.h"
#include "TraceSpan.h"
#include "libkineto.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {

// TODO: Move ConfigDerivedState and its implemenation into separate header and
//       source files.
class Config;

// This struct is a derived snapshot of the Config. And should not
// be mutable after construction.
struct ConfigDerivedState final {
  ConfigDerivedState() = delete;
  ConfigDerivedState(const Config&);

  // Calculate if starting is valid.
  bool canStart(
      const std::chrono::time_point<std::chrono::system_clock>& now) const;

  // TODO: consider using union since only 1 arg is used.
  bool isWarmupDone(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      int64_t currentIter) const;

  bool isCollectionDone(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      int64_t currentIter) const;

  // Set and Get Functions below.
  const std::set<ActivityType>& profileActivityTypes() const {
    return profileActivityTypes_;
  }

  const std::chrono::time_point<std::chrono::system_clock> profileStartTime()
      const {
    return profileStartTime_;
  }

  const std::chrono::time_point<std::chrono::system_clock> profileEndTime()
      const {
    return profileEndTime_;
  }

  const std::chrono::milliseconds profileDuration() const {
    return profileDuration_;
  }

  int64_t profileStartIteration() const {
    return profileStartIter_;
  }
  int64_t profileEndIteration() const {
    return profileEndIter_;
  }
  bool isProfilingByIteration() const {
    return profilingByIter_;
  }

  bool isPerThreadBufferEnabled() const {
    return perThreadBufferEnabled_;
  }

 private:
  std::set<ActivityType> profileActivityTypes_;
  // Start and end time used for triggering and stopping profiling
  std::chrono::time_point<std::chrono::system_clock> profileStartTime_;
  std::chrono::time_point<std::chrono::system_clock> profileEndTime_;
  std::chrono::milliseconds profileDuration_;
  std::chrono::seconds profileWarmupDuration_;
  int64_t profileStartIter_{-1};
  int64_t profileEndIter_{-1};
  bool profilingByIter_{false};
  bool perThreadBufferEnabled_{false};
};

namespace detail {
inline size_t hash_combine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6u) + (seed >> 2u));
}
} // namespace detail

// Correlation flow type for push/pop correlation ID
enum class CorrelationFlowType { Default, User };

class GenericActivityProfiler {
 public:
  GenericActivityProfiler(bool cpuOnly);
  GenericActivityProfiler(const GenericActivityProfiler&) = delete;
  GenericActivityProfiler& operator=(const GenericActivityProfiler&) = delete;
  virtual ~GenericActivityProfiler();

  bool isStopped() const {
    return isGpuCollectionStopped();
  }

  // Collect CPU and GPU traces
  void collectTrace(
      bool collection_done,
      const std::chrono::time_point<std::chrono::system_clock>& now);

  // Used for async requests
  void setLogger(ActivityLogger* logger) {
    logger_ = logger;
  }

  inline void setCpuActivityPresent(bool val) {
    cpuActivityPresent_ = val;
  }

  inline void setGpuActivityPresent(bool val) {
    gpuActivityPresent_ = val;
  }

  inline bool gpuActivityPresent() const {
    return gpuActivityPresent_;
  }

  inline bool traceNonEmpty() const {
    return cpuActivityPresent_ || gpuActivityPresent_;
  }

  void flushWarmupBuffers(
      int64_t currentIter,
      const std::chrono::time_point<std::chrono::system_clock>& nextWakeupTime);

  bool isWarmupDone(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      int64_t currentIter) const {
    return derivedConfig_->isWarmupDone(now, currentIter);
  }

  bool isCollectionDone(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      int64_t currentIter) const {
    return derivedConfig_->isCollectionDone(now, currentIter);
  }

  virtual bool isGpuCollectionStopped() const {
    return false;
  }

  bool isProfilingByIteration() const {
    return derivedConfig_->isProfilingByIteration();
  }

  std::chrono::time_point<std::chrono::system_clock> profileStartTime() const {
    return derivedConfig_->profileStartTime();
  }

  std::chrono::time_point<std::chrono::system_clock> profileEndTime() const {
    return derivedConfig_->profileEndTime();
  }

  int activitiesMaxGpuBufferSizeMB() const {
    return config_ ? config_->activitiesMaxGpuBufferSize() / 1024 / 1024 : 0;
  }

  void startTrace(
      const std::chrono::time_point<std::chrono::system_clock>& now) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    startTraceInternal(now);
  }

  void stopTrace(
      const std::chrono::time_point<std::chrono::system_clock>& now) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    stopTraceInternal(now);
  }

  void cancelTrace(
      const std::chrono::time_point<std::chrono::system_clock>& now) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    stopTraceInternal(now);
    resetInternal();
  }

  void processTrace(ActivityLogger& logger) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    processTraceInternal(logger);
  }

  void completeTrace(ActivityLogger& logger) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    processTraceInternal(logger);
    resetInternal();
  }

  void reset() {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    resetInternal();
  }

  bool canStart(
      const Config& config,
      const std::chrono::time_point<std::chrono::system_clock>& now) const {
    ConfigDerivedState derived(config);
    return derived.canStart(now);
  }

  void configure(
      const Config& config,
      const std::chrono::time_point<std::chrono::system_clock>& now);

  // Toggle GPU tracing during a profile instance
  void toggleCollectionDynamic(const bool enable);

  // Registered with client API to pass CPU trace events over
  void transferCpuTrace(std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace);

  const Config& config() {
    return *config_;
  }

  inline void recordThreadInfo() {
    int32_t sysTid = systemThreadId();
    // Note we're using the lower 32 bits of the (opaque) pthread id
    // as key, because that's what CUPTI records.
    int32_t tid = threadId();
    int32_t pid = processId();
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    recordThreadInfo(sysTid, tid, pid);
  }

  // T107508020: We can deprecate the recordThreadInfo(void) once we optimized
  // profiler_kineto
  void recordThreadInfo(int32_t sysTid, int32_t tid, int32_t pid) {
    if (!resourceInfo_.contains({pid, tid})) {
      resourceInfo_.emplace(
          std::make_pair(pid, tid),
          ResourceInfo{
              .id = sysTid,
              .sortIndex = sysTid,
              .deviceId = pid,
              .name = fmt::format("thread {} ({})", sysTid, getThreadName())});
    }
  }

  void addMetadata(const std::string& key, const std::string& value) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    metadata_[key] = value;
  }

  void addVersionMetadata(const std::string& key, const std::string& value) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    versionMetadata_[key] = value;
  }

  void addChildActivityProfiler(std::unique_ptr<IActivityProfiler> profiler) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    profilers_.push_back(std::move(profiler));
  }

  void pushCorrelationId(uint64_t id);
  void popCorrelationId();

  void pushUserCorrelationId(uint64_t id);
  void popUserCorrelationId();

 protected:
  // Derived classes should be for a particular device, and should override
  // these virtual member functions. We provide empty defaults because
  // GenericActivityProfiler can also be in cpuOnly mode.
  virtual void logGpuVersions() {}
  virtual void setMaxGpuBufferSize([[maybe_unused]] int64_t size) {}
  virtual void enableGpuTracing() {}
  virtual void disableGpuTracing() {}
  virtual void clearGpuActivities() {}
  virtual void processGpuActivities([[maybe_unused]] ActivityLogger& logger) {}
  virtual void synchronizeGpuDevice() {}
  virtual void pushCorrelationIdImpl(
      [[maybe_unused]] uint64_t id,
      [[maybe_unused]] CorrelationFlowType type) {}
  virtual void popCorrelationIdImpl([[maybe_unused]] CorrelationFlowType type) {
  }
  virtual void onResetTraceData() {}
  virtual void onFinalizeTrace(
      [[maybe_unused]] const Config& config,
      [[maybe_unused]] ActivityLogger& logger) {}

  using CpuGpuSpanPair = std::pair<TraceSpan, TraceSpan>;
  static const CpuGpuSpanPair& defaultTraceSpan();

  // Deferred logging of CUDA-event synchronization
  struct DeferredLogEntry {
    uint32_t device;
    uint32_t stream;
    std::function<void()> logMe;
  };

  std::deque<DeferredLogEntry> logQueue_;

  // Map of gpu activities to user defined events
  class GpuUserEventMap {
   public:
    // Insert a user defined event which maps to the gpu trace activity.
    // If the user defined event mapping already exists this will update the
    // gpu side span to include the span of gpuTraceActivity.
    void insertOrExtendEvent(
        const ITraceActivity& cpuTraceActivity,
        const ITraceActivity& gpuTraceActivity);
    // Log out the events to the logger
    void logEvents(ActivityLogger* logger);

    void clear() {
      streamSpanMap_.clear();
    }

   private:
    // device id and stream name
    using StreamKey = std::pair<int64_t, int64_t>;

    // map of correlation id to TraceSpan
    using CorrelationSpanMap =
        std::unordered_map<int64_t, GenericTraceActivity>;
    std::map<StreamKey, CorrelationSpanMap> streamSpanMap_;
  };

  GpuUserEventMap gpuUserEventMap_;
  // id -> activity*
  std::unordered_map<int64_t, const ITraceActivity*> activityMap_;
  // cuda runtime id -> pytorch op id
  // CUPTI provides a mechanism for correlating Cuda events to arbitrary
  // external events, e.g.operator activities from PyTorch.
  std::unordered_map<int64_t, int64_t> cpuCorrelationMap_;
  // CUDA runtime <-> GPU Activity
  std::unordered_map<int64_t, const ITraceActivity*> correlatedCudaActivities_;
  std::unordered_map<int64_t, int64_t> userCorrelationMap_;

  // data structure to collect cuptiActivityFlushAll() latency overhead
  struct profilerOverhead {
    int64_t overhead;
    int cntr;
  };

  void startTraceInternal(
      const std::chrono::time_point<std::chrono::system_clock>& now);

  void stopTraceInternal(
      const std::chrono::time_point<std::chrono::system_clock>& now);

  void processTraceInternal(ActivityLogger& logger);

  void resetInternal();

  void finalizeTrace(const Config& config, ActivityLogger& logger);

  void configureChildProfilers();

  // Process a single CPU trace
  void processCpuTrace(
      libkineto::CpuTraceBuffer& cpuTrace,
      ActivityLogger& logger);

  inline bool hasDeviceResource(int64_t device, int64_t id) {
    return resourceInfo_.contains({device, id});
  }

  // Create resource names for streams
  inline void recordStream(int64_t device, int64_t id, const char* postfix) {
    if (!hasDeviceResource(device, id)) {
      resourceInfo_.emplace(
          std::make_pair(device, id),
          ResourceInfo{
              .id = id,
              .sortIndex = id,
              .deviceId = device,
              .name = fmt::format("stream {} {}", id, postfix)});
    }
  }

  // Create resource names overall for device, id = -1
  inline void recordDevice(int device) {
    constexpr int id = -1;
    if (!hasDeviceResource(device, id)) {
      resourceInfo_.emplace(
          std::make_pair(device, id),
          ResourceInfo{
              .id = id,
              .sortIndex = id,
              .deviceId = device,
              .name = fmt::format("Device {}", device)});
    }
  }

  // Record client trace span for subsequent lookups from activities
  // Also creates a corresponding GPU-side span.
  CpuGpuSpanPair& recordTraceSpan(TraceSpan& span, int gpuOpCount);

  // Returns true if net name is to be tracked for a specified number of
  // iterations.
  bool iterationTargetMatch(libkineto::CpuTraceBuffer& trace);

  // net name to id
  int netId(const std::string& netName);

  const ITraceActivity* linkedActivity(
      int32_t correlationId,
      const std::unordered_map<int64_t, int64_t>& correlationMap);

  const ITraceActivity* cpuActivity(int32_t correlationId);
  void updateGpuNetSpan(const ITraceActivity& gpuOp);
  bool outOfRange(const ITraceActivity& act);
  void handleGpuActivity(const ITraceActivity& act, ActivityLogger* logger);

  void resetTraceData();

  void addOverheadSample(profilerOverhead& counter, int64_t overhead) {
    counter.overhead += overhead;
    counter.cntr++;
  }
  int64_t getOverhead(const profilerOverhead& counter) {
    if (counter.cntr == 0) {
      return 0;
    }
    return counter.overhead / counter.cntr;
  }

  void checkTimestampOrder(const ITraceActivity* act1);

  // On-demand Request Config (should not be modified)
  // TODO: remove this config_, dependency needs to be removed from
  // finalizeTrace.
  std::unique_ptr<const Config> config_;

  // Resolved details about the config and states are stored here.
  std::unique_ptr<ConfigDerivedState> derivedConfig_;

  // Logger used during trace processing
  ActivityLogger* logger_;

  // All recorded trace spans, both CPU and GPU
  // Trace Id -> list of iterations.
  // Using map of lists for the iterator semantics, since we are recording
  // pointers to the elements in this structure.
  std::map<std::string, std::list<CpuGpuSpanPair>> traceSpans_;

  // Maintain a map of client trace activity to trace span.
  // Maps correlation id -> TraceSpan* held by traceSpans_.
  using ActivityTraceMap = std::unordered_map<int64_t, CpuGpuSpanPair*>;
  ActivityTraceMap clientActivityTraceMap_;

  // Cache thread names and system thread ids for pthread ids,
  // and stream ids for GPU streams
  std::map<std::pair<int64_t, int64_t>, ResourceInfo> resourceInfo_;

  std::vector<ActivityLogger::OverheadInfo> overheadInfo_;

  // the overhead to flush the activity buffer
  profilerOverhead flushOverhead_;
  // the overhead to enable/disable activity tracking
  profilerOverhead setupOverhead_;

  bool cpuOnly_{false};
  bool gpuOnly_{false};
  bool cpuActivityPresent_{false};
  bool gpuActivityPresent_{false};

  // Gate for CPU trace ingestion. True only between startTraceInternal()
  // and resetInternal() — spans arriving outside this window are discarded.
  bool acceptCpuTraces_{false};
  std::atomic<bool> toggleState_{true};

  // ***************************************************************************
  // Below state is shared with external threads.
  // These need to either be atomic, accessed under lock or only used
  // by external threads in separate runloop phases from the profiler thread.
  // ***************************************************************************

  // Mutex to protect non-atomic access to below state
  std::recursive_mutex mutex_;

  // Keep track of the start time and end time for the trace collected.
  // External threads using startTrace need to manually stopTrace. Part of the
  // mock tests. All CUDA events before this time will be removed
  int64_t captureWindowStartTime_{0};
  // Similarly, all CUDA API events after the last net event will be removed
  int64_t captureWindowEndTime_{0};

  // span name -> iteration count
  std::map<std::string, int> iterationCountMap_;

  struct DevStream {
    int64_t ctx = 0;
    int64_t stream = 0;
    bool operator==(const DevStream& other) const {
      return (this->ctx == other.ctx) && (this->stream == other.stream);
    }
  };

  struct DevStreamHash {
    std::size_t operator()(const DevStream& c) const {
      return detail::hash_combine(
          std::hash<int64_t>()(c.ctx), std::hash<int64_t>()(c.stream));
    }
  };

  struct ErrorCounts {
    int32_t invalid_external_correlation_events = 0;
    int32_t out_of_range_events = 0;
    int32_t gpu_and_cpu_op_out_of_order = 0;
    int32_t blocklisted_runtime_events = 0;
    int32_t unexepected_cuda_events = 0;
    bool gpu_stopped_early = false;
  };

  friend std::ostream& operator<<(std::ostream& oss, const ErrorCounts& ecs);

  // This set tracks the (device, cuda streams) observed in the trace
  // doing CUDA kernels/memcopies. This prevents emitting CUDA sync
  // events on streams with no activity.
  std::unordered_set<DevStream, DevStreamHash> seenDeviceStreams_;

  // Buffers where trace data is stored
  std::unique_ptr<ActivityBuffers> traceBuffers_;

  // Trace metadata
  std::unordered_map<std::string, std::string> metadata_;

  // Version metadata
  std::unordered_map<std::string, std::string> versionMetadata_;

  // child activity profilers
  std::vector<std::unique_ptr<IActivityProfiler>> profilers_;

  // a vector of active profiler plugin sessions
  std::vector<std::unique_ptr<IActivityProfilerSession>> sessions_;

  // Number of memory overhead events encountered during the session
  uint32_t resourceOverheadCount_;

  ErrorCounts ecs_;
};

} // namespace KINETO_NAMESPACE
