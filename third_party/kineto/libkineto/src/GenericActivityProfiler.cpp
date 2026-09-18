/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "GenericActivityProfiler.h"
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "ApproximateClock.h"

#include "ActivityBuffers.h"
#include "Config.h"
#include "DeviceProperties.h"
#include "DeviceUtil.h"
#include "output_base.h"
#include "time_since_epoch.h"

#include "Logger.h"
#include "ThreadUtil.h"

using namespace std::chrono;
using std::string;

namespace KINETO_NAMESPACE {

// TODO: Move config elsehwere. Sync with @sraikund16 on details.
ConfigDerivedState::ConfigDerivedState(const Config& config) {
  profileActivityTypes_ = config.selectedActivityTypes();
  profileStartTime_ = config.requestTimestamp();
  profileDuration_ = config.activitiesDuration();
  profileWarmupDuration_ = config.activitiesWarmupDuration();
  profilingByIter_ = config.hasProfileStartIteration();
  perThreadBufferEnabled_ = config.perThreadBufferEnabled();
  if (profilingByIter_) {
    profileStartIter_ = config.profileStartIteration();
    profileEndIter_ = profileStartIter_ + config.activitiesRunIterations();
  } else {
    profileEndIter_ = (std::numeric_limits<decltype(profileEndIter_)>::max)();
    profileEndTime_ = profileStartTime_ + config.activitiesDuration();
  }
}

bool ConfigDerivedState::canStart(
    const std::chrono::time_point<std::chrono::system_clock>& now) const {
  if (profilingByIter_) {
    return true;
  }
  if (profileStartTime_ < now) {
    LOG(WARNING)
        << "Not starting tracing - start timestamp is in the past. Time difference (ms): "
        << duration_cast<milliseconds>(now - profileStartTime_).count();
    return false;
  } else if ((profileStartTime_ - now) < profileWarmupDuration_) {
    LOG(WARNING)
        << "Not starting tracing - insufficient time for warmup. Time to warmup (ms): "
        << duration_cast<milliseconds>(profileStartTime_ - now).count();
    return false;
  }
  return true;
}

bool ConfigDerivedState::isCollectionDone(
    const time_point<system_clock>& now,
    int64_t currentIter) const {
  bool isTimestampBased = !profilingByIter_ && currentIter < 0;
  if (isTimestampBased) {
    // qualify that this check is not being called from application step() API
    return now >= profileEndTime_;
  }
  bool isIterationBased = profilingByIter_ && currentIter >= 0;
  if (isIterationBased) {
    return currentIter >= profileEndIter_;
  }
  return false;
}

std::ostream& operator<<(
    std::ostream& oss,
    const GenericActivityProfiler::ErrorCounts& ecs) {
  oss << "Out-of-range = " << ecs.out_of_range_events
      << ", Blocklisted runtime = " << ecs.blocklisted_runtime_events
      << ", Invalid ext correlations = "
      << ecs.invalid_external_correlation_events
      << ", CPU GPU out-of-order = " << ecs.gpu_and_cpu_op_out_of_order
      << ", Unexpected CUDA events = " << ecs.unexepected_cuda_events
      << ", GPU stopped early? = " << ecs.gpu_stopped_early;
  return oss;
}

GenericActivityProfiler::GenericActivityProfiler(bool cpuOnly)
    : flushOverhead_{0, 0}, setupOverhead_{0, 0}, cpuOnly_{cpuOnly} {}
GenericActivityProfiler::~GenericActivityProfiler() {}

void GenericActivityProfiler::transferCpuTrace(
    std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace) {
  std::lock_guard<std::recursive_mutex> guard(mutex_);
  const string& trace_name = cpuTrace->span.name;
  // acceptCpuTraces_ stays true after the synchronous processTrace() path moves
  // traceBuffers_ out via finalizeTrace() (only completeTrace()/resetInternal()
  // clears it), so a late span can arrive with traceBuffers_ already null.
  // Treat either condition as "not collecting" and discard instead of
  // dereferencing.
  if (!acceptCpuTraces_ || traceBuffers_ == nullptr) {
    VLOG(0) << "Trace collection not in progress - discarding span "
            << trace_name;
    return;
  }

  cpuTrace->span.iteration = iterationCountMap_[trace_name]++;

  VLOG(0) << "Received iteration " << cpuTrace->span.iteration << " of span "
          << trace_name << " (" << cpuTrace->activities.size()
          << " activities / " << cpuTrace->gpuOpCount << " gpu activities)";
  traceBuffers_->cpu.push_back(std::move(cpuTrace));
}

namespace {

const std::unordered_set<std::string>& getLoggerMedataAllowList() {
  static const std::unordered_set<std::string> kLoggerMedataAllowList{
      "with_stack", "with_modules", "record_shapes", "profile_memory"};
  return kLoggerMedataAllowList;
}

} // namespace

void GenericActivityProfiler::processTraceInternal(ActivityLogger& logger) {
  UST_LOGGER_STAGE_SCOPE(kPostProcessingStage);
  // A previous processTraceInternal() moved traceBuffers_ out via
  // finalizeTrace(); a redundant process call can reach here with it already
  // null. Bail out instead of dereferencing null.
  if (traceBuffers_ == nullptr) {
    LOG(WARNING) << "No trace buffers to process - skipping";
    return;
  }
  LOG(INFO) << "Processing " << traceBuffers_->cpu.size() << " CPU buffers";
  VLOG(0) << "Profile time range: " << captureWindowStartTime_ << " - "
          << captureWindowEndTime_;

  // Pass metadata within the trace to the logger observer.
  for (const auto& pair : metadata_) {
    if (getLoggerMedataAllowList().contains(pair.first)) {
      LOGGER_OBSERVER_ADD_METADATA(pair.first, pair.second);
    }
  }
  for (auto& pair : versionMetadata_) {
    addMetadata(pair.first, pair.second);
  }
  std::vector<std::string> device_properties;
  if (const auto& props = devicePropertiesJson(); !props.empty()) {
    device_properties.push_back(props);
  }
  for (const auto& session : sessions_) {
    if (auto props = session->getDeviceProperties(); !props.empty()) {
      if (std::ranges::find(device_properties, props) ==
          device_properties.end()) {
        device_properties.push_back(props);
      }
    }
    for (const auto& [key, value] : session->getMetadata()) {
      addMetadata(key, value);
    }
  }
  logger.handleTraceStart(
      metadata_, fmt::format("{}", fmt::join(device_properties, ",")));
  setCpuActivityPresent(false);
  setGpuActivityPresent(false);
  for (auto& cpu_trace : traceBuffers_->cpu) {
    string trace_name = cpu_trace->span.name;
    VLOG(0) << "Processing CPU buffer for " << trace_name << " ("
            << cpu_trace->span.iteration << ") - "
            << cpu_trace->activities.size() << " records";
    VLOG(0) << "Span time range: " << cpu_trace->span.startTime << " - "
            << cpu_trace->span.endTime;
    processCpuTrace(*cpu_trace, logger);
    LOGGER_OBSERVER_ADD_EVENT_COUNT(cpu_trace->activities.size());
  }

  // Process GPU activities via derived class
  if (!cpuOnly_) {
    processGpuActivities(logger);
    if (!gpuActivityPresent()) {
      LOG(WARNING) << "GPU trace is empty!";
    }
  }

  if (!traceNonEmpty()) {
    LOG(WARNING) << kEmptyTrace;
  }

  for (const auto& session : sessions_) {
    LOG(INFO) << "Processing child profiler trace";
    // cpuActivity() function here is used to get the linked cpuActivity for
    // session's activities. Passing captureWindowStartTime_ and
    // captureWindowEndTime_ in order to specify the range of activities that
    // need to be processed.
    session->processTrace(
        logger,
        [this](auto&& correlationId) {
          return cpuActivity(
              std::forward<decltype(correlationId)>(correlationId));
        },
        captureWindowStartTime_,
        captureWindowEndTime_);
  }

  LOG(INFO) << "Record counts: " << ecs_;

  finalizeTrace(*config_, logger);
}

GenericActivityProfiler::CpuGpuSpanPair& GenericActivityProfiler::
    recordTraceSpan(TraceSpan& span, int gpuOpCount) {
  TraceSpan gpu_span(gpuOpCount, span.iteration, span.name, "GPU: ");
  auto& iterations = traceSpans_[span.name];
  iterations.emplace_back(span, gpu_span);
  return iterations.back();
}

void GenericActivityProfiler::processCpuTrace(
    libkineto::CpuTraceBuffer& cpuTrace,
    ActivityLogger& logger) {
  if (cpuTrace.activities.empty()) {
    LOG(WARNING) << "CPU trace is empty!";
    return;
  }
  setCpuActivityPresent(true);
  bool warn_once = false;
  CpuGpuSpanPair& span_pair =
      recordTraceSpan(cpuTrace.span, cpuTrace.gpuOpCount);
  TraceSpan& cpu_span = span_pair.first;
  for (auto const& act : cpuTrace.activities) {
    VLOG(2) << act->correlationId() << ": OP " << act->activityName;
    if (derivedConfig_->profileActivityTypes().contains(act->type())) {
      static_assert(
          std::is_same_v<
              std::remove_reference_t<decltype(act)>,
              const std::unique_ptr<GenericTraceActivity>>,
          "handleActivity is unsafe and relies on the caller to maintain not "
          "only lifetime but also address stability.");
      if (act->duration() < 0) {
        act->endTime = captureWindowEndTime_;
        act->addMetadata("finished", "false");
      }
      logger.handleActivity(*act);
    }
    clientActivityTraceMap_[act->correlationId()] = &span_pair;
    activityMap_[act->correlationId()] = act.get();
    if (act->deviceId() == 0) {
      if (!warn_once) {
        LOG(WARNING)
            << "CPU activity with pid 0 detected. This is likely due to the python stack"
               " tracer not being able to determine the pid for an event. Overriding pid to main thread pid";
      }
      act->setDevice(processId());
      warn_once = true;
    }
    recordThreadInfo(act->resourceId(), act->getThreadId(), act->deviceId());
  }
  logger.handleTraceSpan(cpu_span);
}

static GenericTraceActivity createUserGpuSpan(
    const libkineto::ITraceActivity& cpuTraceActivity,
    const libkineto::ITraceActivity& gpuTraceActivity) {
  GenericTraceActivity res(
      *cpuTraceActivity.traceSpan(),
      ActivityType::GPU_USER_ANNOTATION,
      cpuTraceActivity.name());
  res.startTime = gpuTraceActivity.timestamp();
  res.device = gpuTraceActivity.deviceId();
  res.resource = gpuTraceActivity.resourceId();
  res.endTime = gpuTraceActivity.timestamp() + gpuTraceActivity.duration();
  res.id = cpuTraceActivity.correlationId();
  return res;
}

void GenericActivityProfiler::GpuUserEventMap::insertOrExtendEvent(
    const ITraceActivity& cpuTraceActivity,
    const ITraceActivity& gpuTraceActivity) {
  StreamKey key(gpuTraceActivity.deviceId(), gpuTraceActivity.resourceId());
  CorrelationSpanMap& correlationSpanMap = streamSpanMap_[key];
  auto it = correlationSpanMap.find(cpuTraceActivity.correlationId());
  if (it == correlationSpanMap.end()) {
    auto it_success = correlationSpanMap.insert(
        {cpuTraceActivity.correlationId(),
         createUserGpuSpan(cpuTraceActivity, gpuTraceActivity)});
    it = it_success.first;
  }
  GenericTraceActivity& span = it->second;
  if (gpuTraceActivity.timestamp() < span.startTime || span.startTime == 0) {
    span.startTime = gpuTraceActivity.timestamp();
  }
  int64_t gpu_activity_end =
      gpuTraceActivity.timestamp() + gpuTraceActivity.duration();
  span.endTime = std::max(gpu_activity_end, span.endTime);
}

const GenericActivityProfiler::CpuGpuSpanPair& GenericActivityProfiler::
    defaultTraceSpan() {
  static TraceSpan span(0, 0, "Unknown", "");
  static CpuGpuSpanPair span_pair(span, span);
  return span_pair;
}

void GenericActivityProfiler::GpuUserEventMap::logEvents(
    ActivityLogger* logger) {
  for (auto const& streamMapPair : streamSpanMap_) {
    for (auto const& correlationSpanPair : streamMapPair.second) {
      correlationSpanPair.second.log(*logger);
    }
  }
}

bool GenericActivityProfiler::outOfRange(const ITraceActivity& act) {
  bool out_of_range = act.timestamp() < captureWindowStartTime_ ||
      (act.timestamp() + act.duration()) > captureWindowEndTime_;
  if (out_of_range) {
    VLOG(2) << "TraceActivity outside of profiling window: " << act.name()
            << " (" << act.timestamp() << " < " << captureWindowStartTime_
            << " or " << (act.timestamp() + act.duration()) << " > "
            << captureWindowEndTime_;
    ecs_.out_of_range_events++;
  }
  return out_of_range;
}

inline void GenericActivityProfiler::updateGpuNetSpan(
    const ITraceActivity& gpuOp) {
  if (!gpuOp.linkedActivity()) {
    VLOG(0) << "Missing linked activity";
    return;
  }
  const auto& it =
      clientActivityTraceMap_.find(gpuOp.linkedActivity()->correlationId());
  if (it == clientActivityTraceMap_.end()) {
    // No correlation id mapping?
    return;
  }
  TraceSpan& gpu_span = it->second->second;
  if (gpuOp.timestamp() < gpu_span.startTime || gpu_span.startTime == 0) {
    gpu_span.startTime = gpuOp.timestamp();
  }
  gpu_span.endTime =
      std::max(gpuOp.timestamp() + gpuOp.duration(), gpu_span.endTime);
}

// I've observed occasional broken timestamps attached to GPU events...
void GenericActivityProfiler::checkTimestampOrder(const ITraceActivity* act1) {
  // Correlated GPU runtime activity cannot
  // have timestamp greater than the GPU activity's
  const auto& it = correlatedCudaActivities_.find(act1->correlationId());
  if (it == correlatedCudaActivities_.end()) {
    correlatedCudaActivities_.insert({act1->correlationId(), act1});
    return;
  }

  // Activities may be appear in the buffers out of order.
  // If we have a runtime activity in the map, it should mean that we
  // have a GPU activity passed in, and vice versa.
  const ITraceActivity* act2 = it->second;
  if (act2->type() == ActivityType::CUDA_RUNTIME) {
    // Buffer is out-of-order.
    // Swap so that runtime activity is first for the comparison below.
    std::swap(act1, act2);
  }
  // Range Profiling mode returns kernels with 0 ts and duration that we can
  // pass through to output
  if (act2->timestamp() == 0) {
    return;
  }
  if (act1->timestamp() > act2->timestamp()) {
    LOG_FIRST_N(WARNING, 10)
        << "GPU op timestamp (" << act2->timestamp()
        << ") < runtime timestamp (" << act1->timestamp() << ") by "
        << act1->timestamp() - act2->timestamp() << "us"
        << " Name: " << act2->name() << " Device: " << act2->deviceId()
        << " Stream: " << act2->resourceId();
    ecs_.gpu_and_cpu_op_out_of_order++;
  }
}

const ITraceActivity* GenericActivityProfiler::linkedActivity(
    int32_t correlationId,
    const std::unordered_map<int64_t, int64_t>& correlationMap) {
  const auto& it = correlationMap.find(correlationId);
  if (it != correlationMap.end()) {
    const auto& it2 = activityMap_.find(it->second);
    if (it2 != activityMap_.end()) {
      return it2->second;
    }
  }
  return nullptr;
}

void GenericActivityProfiler::handleGpuActivity(
    const ITraceActivity& act,
    ActivityLogger* logger) {
  if (outOfRange(act)) {
    return;
  }
  checkTimestampOrder(&act);
  VLOG(2) << act.correlationId() << ": " << act.name();
  recordStream(act.deviceId(), act.resourceId(), "");
  seenDeviceStreams_.insert({act.deviceId(), act.resourceId()});

  act.log(*logger);
  setGpuActivityPresent(true);
  updateGpuNetSpan(act);
  if (derivedConfig_->profileActivityTypes().contains(
          ActivityType::GPU_USER_ANNOTATION)) {
    const auto& it = userCorrelationMap_.find(act.correlationId());
    if (it != userCorrelationMap_.end()) {
      const auto& it2 = activityMap_.find(it->second);
      if (it2 != activityMap_.end()) {
        recordStream(act.deviceId(), act.resourceId(), "context");
        gpuUserEventMap_.insertOrExtendEvent(*it2->second, act);
      }
    }
  }
}

const ITraceActivity* GenericActivityProfiler::cpuActivity(
    int32_t correlationId) {
  const auto& it2 = activityMap_.find(correlationId);
  return (it2 != activityMap_.end()) ? it2->second : nullptr;
}

void GenericActivityProfiler::configureChildProfilers() {
  // If child profilers are enabled create profiler sessions
  int64_t start_time_ms =
      duration_cast<milliseconds>(
          derivedConfig_->profileStartTime().time_since_epoch())
          .count();
  for (auto& profiler : profilers_) {
    LOG(INFO) << "[Profiler = " << profiler->name() << "] "
              << "Evaluating whether to run child profiler.";
    auto session = profiler->configure(
        start_time_ms,
        derivedConfig_->profileDuration().count(),
        derivedConfig_->profileActivityTypes(),
        *config_);
    if (session) {
      LOG(INFO) << "[Profiler = " << profiler->name() << "] "
                << "Running child profiler " << profiler->name() << " for "
                << derivedConfig_->profileDuration().count() << " ms";
      sessions_.push_back(std::move(session));
    } else {
      LOG(INFO) << "[Profiler = " << profiler->name() << "] "
                << "Not running child profiler.";
    }
  }
}

void GenericActivityProfiler::configure(
    const Config& config,
    const time_point<system_clock>& now) {
  std::lock_guard<std::recursive_mutex> guard(mutex_);
  ApproximateClockToUnixTimeConverter clockConverter;
  get_time_converter() = clockConverter.makeConverter();

  config_ = config.clone();

  // Ensure we're starting in a clean state
  resetTraceData();

  derivedConfig_.reset();
  derivedConfig_ = std::make_unique<ConfigDerivedState>(*config_);

  if (LOG_IS_ON(INFO)) {
    config_->printActivityProfilerConfig(LIBKINETO_DBG_STREAM);
  }
  if (!cpuOnly_ && !libkineto::api().client()) {
    gpuOnly_ = true;
    if (derivedConfig_->isProfilingByIteration()) {
      LOG(INFO) << "GPU-only tracing for " << config_->activitiesRunIterations()
                << " iterations";
    } else {
      LOG(INFO) << "GPU-only tracing for "
                << config_->activitiesDuration().count() << "ms";
    }
  }

  // Set useful metadata into the logger.
  LOGGER_OBSERVER_SET_TRACE_DURATION_MS(config_->activitiesDuration().count());
  LOGGER_OBSERVER_SET_TRACE_ID(config_->requestTraceID());
  LOGGER_OBSERVER_SET_GROUP_TRACE_ID(config_->requestGroupTraceID());
  if (!config_->requestTraceID().empty()) {
    addMetadata("trace_id", "\"" + config_->requestTraceID() + "\"");
  }

  if (!cpuOnly_) {
    // Enabling CUPTI activity tracing incurs a larger perf hit at first,
    // presumably because structures are allocated and initialized, callbacks
    // are activated etc. After a while the overhead decreases and stabilizes.
    // It's therefore useful to perform some warmup before starting recording.
    LOG(INFO) << "Enabling GPU tracing with max buffer size "
              << config_->activitiesMaxGpuBufferSize() / 1024 / 1024 << "MB)";
    setMaxGpuBufferSize(config_->activitiesMaxGpuBufferSize());
    time_point<system_clock> timestamp;
    if (VLOG_IS_ON(1)) {
      timestamp = system_clock::now();
    }
    toggleState_.store(true);
    enableGpuTracing();
    if (VLOG_IS_ON(1)) {
      auto t2 = system_clock::now();
      addOverheadSample(
          setupOverhead_, duration_cast<microseconds>(t2 - timestamp).count());
    }
  }

  if (!profilers_.empty()) {
    configureChildProfilers();
  }

  if (libkineto::api().client()) {
    libkineto::api().client()->prepare(
        config_->isReportInputShapesEnabled(),
        config_->isProfileMemoryEnabled(),
        config_->isWithStackEnabled(),
        config_->isWithFlopsEnabled(),
        config_->isWithModulesEnabled());
  }

  if (derivedConfig_->isProfilingByIteration()) {
    LOG(INFO) << "Tracing starting on iteration = "
              << derivedConfig_->profileStartIteration();
    LOG(INFO) << "Tracing will end on iteration = "
              << derivedConfig_->profileEndIteration();
  } else {
    LOG(INFO) << "Tracing starting in "
              << duration_cast<seconds>(
                     derivedConfig_->profileStartTime() - now)
                     .count()
              << "s";
    LOG(INFO) << "Tracing will end in "
              << duration_cast<seconds>(derivedConfig_->profileEndTime() - now)
                     .count()
              << "s";
  }

  traceBuffers_ = std::make_unique<ActivityBuffers>();
  captureWindowStartTime_ = captureWindowEndTime_ = 0;
  acceptCpuTraces_ = false;
}

void GenericActivityProfiler::flushWarmupBuffers(
    int64_t currentIter,
    const time_point<system_clock>& nextWakeupTime) {
  if (cpuOnly_) {
    return;
  }
  // Flushing GPU activities can take a while so avoid doing it close to
  // the start time. Clear during warmup in the following cases:
  // 1. Iteration-based flow: called from application step() API
  //    (currentIter >= 0) with iteration profiling enabled
  // 2. Timestamp-based flow: called from periodic runloop
  //    (currentIter < 0) when not close to profile start time
  // 3. Iteration config with periodic runloop: always safe to clear
  const bool isIterationBasedFlow =
      derivedConfig_->isProfilingByIteration() && currentIter >= 0;
  const bool isTimestampBasedFlowSafeToFlush =
      !derivedConfig_->isProfilingByIteration() && currentIter < 0 &&
      nextWakeupTime < derivedConfig_->profileStartTime();
  const bool isIterationConfigWithPeriodicRunloop =
      derivedConfig_->isProfilingByIteration() && currentIter < 0;

  if (isIterationBasedFlow || isTimestampBasedFlowSafeToFlush ||
      isIterationConfigWithPeriodicRunloop) {
    clearGpuActivities();
  }
}

void GenericActivityProfiler::toggleCollectionDynamic(const bool enable) {
  if (toggleState_.load() == enable) {
    return;
  }
  toggleState_.store(enable);

  // The ordering of conditions and synchronization is deliberate. We
  // intentionally synchronize:
  //
  //   - AFTER we disable
  //   - BEFORE we enable
  //
  // Both to prevent the synchronization event from showing up in traces.
  if (!enable) {
    for (auto& session : sessions_) {
      session->toggleCollectionDynamic(enable);
    }
    disableGpuTracing();
  }
  synchronizeGpuDevice();
  if (enable) {
    enableGpuTracing();
    for (auto& session : sessions_) {
      session->toggleCollectionDynamic(enable);
    }
  }
}

void GenericActivityProfiler::startTraceInternal(
    const time_point<system_clock>& now) {
  captureWindowStartTime_ = libkineto::timeSinceEpoch(now);
  for (auto& session : sessions_) {
    LOG(INFO) << "Starting child profiler session";
    session->start();
  }
  acceptCpuTraces_ = true;
}

void GenericActivityProfiler::stopTraceInternal(
    const time_point<system_clock>& now) {
  captureWindowEndTime_ = libkineto::timeSinceEpoch(now);
  if (!cpuOnly_) {
    time_point<system_clock> timestamp;
    if (VLOG_IS_ON(1)) {
      timestamp = system_clock::now();
    }
    toggleState_.store(false);
    disableGpuTracing();
    if (VLOG_IS_ON(1)) {
      auto t2 = system_clock::now();
      addOverheadSample(
          setupOverhead_, duration_cast<microseconds>(t2 - timestamp).count());
    }
  }

  for (auto& session : sessions_) {
    LOG(INFO) << "Stopping child profiler session";
    session->stop();
  }
}

void GenericActivityProfiler::resetInternal() {
  acceptCpuTraces_ = false;
  resetTraceData();
}

void GenericActivityProfiler::finalizeTrace(
    const Config& config,
    ActivityLogger& logger) {
  LOG(INFO) << "CPU Traces Recorded:";
  {
    for (const auto& it : iterationCountMap_) {
      LOG(INFO) << it.first << ": " << it.second << " span(s) recorded";
    }
    iterationCountMap_.clear();
  }

  // Thread & stream info
  for (const auto& pair : resourceInfo_) {
    const auto& resource = pair.second;
    logger.handleResourceInfo(resource, captureWindowStartTime_);
  }

  bool use_default_device_info = true;
  for (auto& session : sessions_) {
    auto device_info = session->getDeviceInfo();
    if (device_info != nullptr) {
      use_default_device_info = false;
      logger.handleDeviceInfo(*device_info, captureWindowStartTime_);
    }

    auto resource_infos = session->getResourceInfos();
    for (const auto& resource_info : resource_infos) {
      logger.handleResourceInfo(resource_info, captureWindowStartTime_);
    }
  }

  // Process names
  int32_t pid = processId();
  string process_name = processName(pid);
  if (!process_name.empty()) {
    logger.handleDeviceInfo(
        {.id = pid, .sortIndex = pid, .name = process_name, .label = "CPU"},
        captureWindowStartTime_);
    if (!cpuOnly_ && use_default_device_info) {
      // Usually, GPU events use device id as pid (0-7).
      // In some cases, CPU sockets are numbered starting from 0.
      // In the worst case, 8 CPU sockets + 8 GPUs, so the max GPU ID is 15.
      //
      // TODO: We should either use a defined constant from Cupti or make an
      //       API call to get this value.
      constexpr int kMaxGpuID = 15;
      // sortIndex is gpu + kExceedMaxPid to put GPU tracks at the bottom
      // of the trace timelines.
      for (int gpu = 0; gpu <= kMaxGpuID; gpu++) {
        logger.handleDeviceInfo(
            {.id = gpu,
             .sortIndex = gpu + kExceedMaxPid,
             .name = process_name,
             .label = fmt::format("GPU {}", gpu)},
            captureWindowStartTime_);
      }
    }
  }

  for (const auto& iterations : traceSpans_) {
    for (const auto& span_pair : iterations.second) {
      const TraceSpan& gpu_span = span_pair.second;
      if (gpu_span.opCount > 0) {
        logger.handleTraceSpan(gpu_span);
      }
    }
  }

  // Call derived class hook for device-specific finalization
  onFinalizeTrace(config, logger);

  gpuUserEventMap_.logEvents(&logger);

  for (auto& session : sessions_) {
    auto trace_buffer = session->getTraceBuffer();
    if (trace_buffer) {
      // Set child start time to profiling start time if not set
      if (trace_buffer->span.startTime == 0) {
        trace_buffer->span.startTime = captureWindowStartTime_;
      }
      traceBuffers_->cpu.push_back(std::move(trace_buffer));
    }
  }

  logger.finalizeTrace(config, std::move(traceBuffers_), captureWindowEndTime_);
}

void GenericActivityProfiler::pushCorrelationId(uint64_t id) {
  pushCorrelationIdImpl(id, CorrelationFlowType::Default);
  for (auto& session : sessions_) {
    session->pushCorrelationId(id);
  }
}

void GenericActivityProfiler::popCorrelationId() {
  popCorrelationIdImpl(CorrelationFlowType::Default);
  for (auto& session : sessions_) {
    session->popCorrelationId();
  }
}

void GenericActivityProfiler::pushUserCorrelationId(uint64_t id) {
  pushCorrelationIdImpl(id, CorrelationFlowType::User);
  for (auto& session : sessions_) {
    session->pushUserCorrelationId(id);
  }
}

void GenericActivityProfiler::popUserCorrelationId() {
  popCorrelationIdImpl(CorrelationFlowType::User);
  for (auto& session : sessions_) {
    session->popUserCorrelationId();
  }
}

void GenericActivityProfiler::resetTraceData() {
  if (!cpuOnly_) {
    clearGpuActivities();
    onResetTraceData();
  }
  activityMap_.clear();
  cpuCorrelationMap_.clear();
  correlatedCudaActivities_.clear();
  gpuUserEventMap_.clear();
  traceSpans_.clear();
  clientActivityTraceMap_.clear();
  seenDeviceStreams_.clear();
  logQueue_.clear();
  traceBuffers_ = nullptr;
  metadata_.clear();
  sessions_.clear();
  resourceOverheadCount_ = 0;
  ecs_ = ErrorCounts{};
}

void GenericActivityProfiler::collectTrace(
    bool collection_done,
    const std::chrono::time_point<std::chrono::system_clock>& now) {
  if (libkineto::api().client()) {
    libkineto::api().client()->stop();
  }

  if (isGpuCollectionStopped()) {
    ecs_.gpu_stopped_early = true;
    LOG(ERROR)
        << "State: CollectTrace stopped by GPU profiler. (Buffer size configured is "
        << config_->activitiesMaxGpuBufferSize() / 1024 / 1024 << "MB)";
  }
  std::lock_guard<std::recursive_mutex> guard(mutex_);
  stopTraceInternal(now);
  VLOG_IF(0, collection_done) << "Reached profile end time";
  UST_LOGGER_MARK_COMPLETED(kCollectionStage);
}

bool ConfigDerivedState::isWarmupDone(
    const time_point<system_clock>& now,
    int64_t currentIter) const {
  bool isTimestampBased = !profilingByIter_ && currentIter < 0;
  if (isTimestampBased) {
    // qualify that this check is not being called from application step() API
    // this avoids races between the step() API and periodically invoked
    // profiler run loop step() method
    return now >= profileStartTime_;
  }
  bool isIterationBased = profilingByIter_ && currentIter >= 0;
  if (isIterationBased) {
    return currentIter >= profileStartIter_;
  }
  return false;
}

} // namespace KINETO_NAMESPACE
