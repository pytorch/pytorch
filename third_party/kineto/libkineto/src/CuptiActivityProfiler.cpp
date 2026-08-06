/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiActivityProfiler.h"
#include <cupti.h>
#include <fmt/format.h>
#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ActivityBuffers.h"
#include "Config.h"
#include "CuptiActivity.h"
#include "CuptiActivityApi.h"
#include "CuptiCbidRegistry.h"
#include "Demangle.h"
#include "DeviceUtil.h"
#include "KernelRegistry.h"
#include "Logger.h"

using namespace std::chrono;
using std::string;

namespace {

struct CtxEventPair {
  uint32_t ctx = 0;
  uint32_t eventId = 0;

  bool operator==(const CtxEventPair& other) const {
    return (this->ctx == other.ctx) && (this->eventId == other.eventId);
  }
};

struct CtxEventPairHash {
  std::size_t operator()(const CtxEventPair& c) const {
    return KINETO_NAMESPACE::detail::hash_combine(
        std::hash<uint32_t>()(c.ctx), std::hash<uint32_t>()(c.eventId));
  }
};

struct WaitEventInfo {
  // CUDA stream that the CUDA event was recorded on
  uint32_t stream;
  // Correlation ID of the cudaEventRecord event
  uint32_t correlationId;
};

// Map (ctx, eventId) -> list of WaitEventInfo for all recorded events.
std::unordered_map<CtxEventPair, std::vector<WaitEventInfo>, CtxEventPairHash>&
waitEventMap() {
  static std::
      unordered_map<CtxEventPair, std::vector<WaitEventInfo>, CtxEventPairHash>
          waitEventMap_;
  return waitEventMap_;
}

// Map ctx -> deviceId
std::unordered_map<uint32_t, uint32_t>& ctxToDeviceId() {
  static std::unordered_map<uint32_t, uint32_t> ctxToDeviceId_;
  return ctxToDeviceId_;
}

} // namespace

namespace KINETO_NAMESPACE {

bool& use_cupti_tsc() {
  static bool use_cupti_tsc = true;
  return use_cupti_tsc;
}

CuptiActivityProfiler::CuptiActivityProfiler(
    CuptiActivityApi& cupti,
    bool cpuOnly)
    : GenericActivityProfiler(cpuOnly), cupti_(cupti) {
  if (isGpuAvailable()) {
    logGpuVersions();
  }
}

void CuptiActivityProfiler::logGpuVersions() {
  uint32_t cuptiVersion = 0;
  int cudaRuntimeVersion = 0;
  int cudaDriverVersion = 0;
  CUPTI_CALL(cuptiGetVersion(&cuptiVersion));
  CUDA_CALL(cudaRuntimeGetVersion(&cudaRuntimeVersion));
  CUDA_CALL(cudaDriverGetVersion(&cudaDriverVersion));
  LOG(INFO) << "CUDA versions. CUPTI: " << cuptiVersion
            << "; Runtime: " << cudaRuntimeVersion
            << "; Driver: " << cudaDriverVersion;

  LOGGER_OBSERVER_ADD_PERSISTENT_METADATA(
      "cupti_version", std::to_string(cuptiVersion));
  LOGGER_OBSERVER_ADD_PERSISTENT_METADATA(
      "cuda_runtime_version", std::to_string(cudaRuntimeVersion));
  LOGGER_OBSERVER_ADD_PERSISTENT_METADATA(
      "cuda_driver_version", std::to_string(cudaDriverVersion));
  addVersionMetadata("cupti_version", std::to_string(cuptiVersion));
  addVersionMetadata(
      "cuda_runtime_version", std::to_string(cudaRuntimeVersion));
  addVersionMetadata("cuda_driver_version", std::to_string(cudaDriverVersion));
}

void CuptiActivityProfiler::setMaxGpuBufferSize(int64_t size) {
  cupti_.setMaxBufferSize(size);
}

void CuptiActivityProfiler::enableGpuTracing() {
#ifdef _WIN32
  CUPTI_CALL(cuptiActivityRegisterTimestampCallback([]() -> uint64_t {
    auto system = std::chrono::time_point_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now());
    return system.time_since_epoch().count();
  }));
#else
#if CUDA_VERSION >= 11060
  use_cupti_tsc() = config().getTSCTimestampFlag();
  if (use_cupti_tsc()) {
    CUPTI_CALL(cuptiActivityRegisterTimestampCallback(
        []() -> uint64_t { return getApproximateTime(); }));
  }
#endif // CUDA_VERSION >= 11060
#endif // _WIN32
  cupti_.enableCuptiActivities(
      derivedConfig_->profileActivityTypes(),
      derivedConfig_->isPerThreadBufferEnabled());
}

void CuptiActivityProfiler::disableGpuTracing() {
  cupti_.disableCuptiActivities(derivedConfig_->profileActivityTypes());
}

void CuptiActivityProfiler::clearGpuActivities() {
  cupti_.clearActivities();
}

bool CuptiActivityProfiler::isGpuCollectionStopped() const {
  return cupti_.stopCollection;
}

void CuptiActivityProfiler::synchronizeGpuDevice() {
  CUDA_CALL(cudaDeviceSynchronize());
  cupti_.flushActivities();
}

void CuptiActivityProfiler::pushCorrelationIdImpl(
    uint64_t id,
    CorrelationFlowType type) {
  CuptiActivityApi::CorrelationFlowType cuptiType =
      (type == CorrelationFlowType::User)
      ? CuptiActivityApi::CorrelationFlowType::User
      : CuptiActivityApi::CorrelationFlowType::Default;
  CuptiActivityApi::pushCorrelationID(id, cuptiType);
}

void CuptiActivityProfiler::popCorrelationIdImpl(CorrelationFlowType type) {
  CuptiActivityApi::CorrelationFlowType cuptiType =
      (type == CorrelationFlowType::User)
      ? CuptiActivityApi::CorrelationFlowType::User
      : CuptiActivityApi::CorrelationFlowType::Default;
  CuptiActivityApi::popCorrelationID(cuptiType);
}

void CuptiActivityProfiler::onResetTraceData() {
  cupti_.teardownContext();
  KernelRegistry::singleton()->clear();
  waitEventMap().clear();
  ctxToDeviceId().clear();
}

void CuptiActivityProfiler::onFinalizeTrace(
    [[maybe_unused]] const Config& config,
    ActivityLogger& logger) {
  // Overhead info
  overheadInfo_.emplace_back("CUPTI Overhead");
  for (const auto& info : overheadInfo_) {
    logger.handleOverheadInfo(info, captureWindowStartTime_);
  }
}

void CuptiActivityProfiler::processGpuActivities(ActivityLogger& logger) {
  VLOG(0) << "Retrieving GPU activity buffers";
  traceBuffers_->gpu = cupti_.activityBuffers();
  if (VLOG_IS_ON(1)) {
    addOverheadSample(flushOverhead_, cupti_.flushOverhead);
  }
  if (traceBuffers_->gpu) {
    // Pass 1: Preprocess all raw records to populate correlation, event, and
    // context lookup state.
    buildProcessingState(*traceBuffers_->gpu);

    // Pass 2: Materialize activities. buildProcessingState() has already
    // populated correlation, event, and context lookup state;
    // EXTERNAL_CORRELATION is a no-op in handleCuptiActivity.
    const auto count_and_size = cupti_.processActivities(
        *traceBuffers_->gpu, [this, &logger](const CUpti_Activity* record) {
          handleCuptiActivity(record, &logger);
        });
    logDeferredEvents();
    LOG(INFO) << "Processed " << count_and_size.first << " GPU records ("
              << count_and_size.second << " bytes)";
    LOGGER_OBSERVER_ADD_EVENT_COUNT(count_and_size.first);

    // resourceOverheadCount_ is set while processing GPU activities
    if (resourceOverheadCount_ > 0) {
      LOG(INFO) << "Allocated " << resourceOverheadCount_
                << " extra CUPTI buffers.";
    }
    LOGGER_OBSERVER_ADD_METADATA(
        "ResourceOverhead", std::to_string(resourceOverheadCount_));
  }
}

// Populate ctxToDeviceId from a record with (contextId, deviceId) fields.
template <class T>
static inline void updateCtxToDeviceId(const T* act) {
  if (!ctxToDeviceId().contains(act->contextId)) {
    ctxToDeviceId()[act->contextId] = act->deviceId;
  }
}

// P2P memcpy carries separate source and destination (context, device) pairs.
static inline void updateCtxToDeviceId(
    const CUpti_ActivityMemcpyPtoPType* act) {
  if (!ctxToDeviceId().contains(act->srcContextId)) {
    ctxToDeviceId()[act->srcContextId] = act->srcDeviceId;
  }
  if (!ctxToDeviceId().contains(act->dstContextId)) {
    ctxToDeviceId()[act->dstContextId] = act->dstDeviceId;
  }
}

// Insert a CUDA event record into waitEventMap(), keeping the per-key vector
// sorted by correlationId so getWaitEventInfo() can binary-search it.
static inline void updateWaitEventMap(const CUpti_ActivityCudaEventType* act) {
  auto key = CtxEventPair{act->contextId, act->eventId};
  std::vector<WaitEventInfo>& vec = waitEventMap()[key];
  auto pos = std::lower_bound(
      vec.begin(),
      vec.end(),
      act->correlationId,
      [](const WaitEventInfo& info, uint32_t val) {
        return info.correlationId < val;
      });
  vec.insert(pos, WaitEventInfo{act->streamId, act->correlationId});
}

void CuptiActivityProfiler::buildProcessingState(
    CuptiActivityBufferMap& buffers) {
  cupti_.processActivities(buffers, [this](const CUpti_Activity* record) {
    switch (record->kind) {
      case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
        handleCorrelationActivity(
            reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(record));
        break;
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        updateCtxToDeviceId(
            reinterpret_cast<const CUpti_ActivityKernelType*>(record));
        break;
      case CUPTI_ACTIVITY_KIND_MEMCPY:
        updateCtxToDeviceId(
            reinterpret_cast<const CUpti_ActivityMemcpyType*>(record));
        break;
      case CUPTI_ACTIVITY_KIND_MEMSET:
        updateCtxToDeviceId(
            reinterpret_cast<const CUpti_ActivityMemsetType*>(record));
        break;
      case CUPTI_ACTIVITY_KIND_MEMCPY2:
        updateCtxToDeviceId(
            reinterpret_cast<const CUpti_ActivityMemcpyPtoPType*>(record));
        break;
      case CUPTI_ACTIVITY_KIND_CUDA_EVENT:
        updateWaitEventMap(
            reinterpret_cast<const CUpti_ActivityCudaEventType*>(record));
        break;
      default:
        break;
    }
  });
}

inline void CuptiActivityProfiler::handleCorrelationActivity(
    const CUpti_ActivityExternalCorrelation* correlation) {
  if (correlation->externalKind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) {
    cpuCorrelationMap_[correlation->correlationId] = correlation->externalId;
  } else if (
      correlation->externalKind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1) {
    userCorrelationMap_[correlation->correlationId] = correlation->externalId;
  } else {
    LOG(WARNING) << "Invalid CUpti_ActivityExternalCorrelation sent to "
                    "handleCuptiActivity";
    ecs_.invalid_external_correlation_events++;
  }
}

void CuptiActivityProfiler::handleRuntimeActivity(
    const CUpti_ActivityAPI* activity,
    ActivityLogger* logger) {
  if (CuptiCbidRegistry::instance().isBlocklisted(
          CallbackDomain::RUNTIME, activity->cbid)) {
    ecs_.blocklisted_runtime_events++;
    return;
  }
  VLOG(2) << activity->correlationId
          << ": CUPTI_ACTIVITY_KIND_RUNTIME, cbid=" << activity->cbid
          << " tid=" << activity->threadId;
  int32_t tid = activity->threadId;
  const auto& it = resourceInfo_.find({processId(), tid});
  if (it != resourceInfo_.end()) {
    tid = it->second.id;
  }
  const ITraceActivity* linked =
      linkedActivity(activity->correlationId, cpuCorrelationMap_);
  const auto& runtime_activity =
      traceBuffers_->addActivityWrapper(RuntimeActivity(activity, linked, tid));
  checkTimestampOrder(&runtime_activity);
  if (outOfRange(runtime_activity)) {
    return;
  }
  runtime_activity.log(*logger);
  setGpuActivityPresent(true);
}

void CuptiActivityProfiler::handleDriverActivity(
    const CUpti_ActivityAPI* activity,
    ActivityLogger* logger) {
  // we only want to collect cuLaunchKernel events, for triton kernel launches
  if (!isTrackedDriverCbid(*activity)) {
    // XXX should we count other driver events?
    return;
  }
  VLOG(2) << activity->correlationId
          << ": CUPTI_ACTIVITY_KIND_DRIVER, cbid=" << activity->cbid
          << " tid=" << activity->threadId;
  int32_t tid = activity->threadId;
  const auto& it = resourceInfo_.find({processId(), tid});
  if (it != resourceInfo_.end()) {
    tid = it->second.id;
  }
  const ITraceActivity* linked =
      linkedActivity(activity->correlationId, cpuCorrelationMap_);
  const auto& runtime_activity =
      traceBuffers_->addActivityWrapper(DriverActivity(activity, linked, tid));
  checkTimestampOrder(&runtime_activity);
  if (outOfRange(runtime_activity)) {
    return;
  }
  runtime_activity.log(*logger);
  setGpuActivityPresent(true);
}

void CuptiActivityProfiler::handleOverheadActivity(
    const CUpti_ActivityOverhead* activity,
    ActivityLogger* logger) {
  VLOG(2) << ": CUPTI_ACTIVITY_KIND_OVERHEAD"
          << " overheadKind=" << activity->overheadKind;
  const auto& overhead_activity =
      traceBuffers_->addActivityWrapper(OverheadActivity(activity, nullptr));
  // Monitor memory overhead
  if (activity->overheadKind == CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE) {
    resourceOverheadCount_++;
  }

  if (outOfRange(overhead_activity)) {
    return;
  }
  overhead_activity.log(*logger);
  setGpuActivityPresent(true);
}

static std::optional<WaitEventInfo> getWaitEventInfo(
    uint32_t ctx,
    uint32_t eventId,
    uint32_t queryCorrelationId) {
  auto key = CtxEventPair{ctx, eventId};
  auto it = waitEventMap().find(key);
  if (it != waitEventMap().end()) {
    // Records are sorted by correlationId. Find the most recent record
    // that precedes the query (i.e., the last record with
    // correlationId < queryCorrelationId).
    const std::vector<WaitEventInfo>& vec = it->second;
    std::vector<WaitEventInfo>::const_iterator pos = std::upper_bound(
        vec.begin(),
        vec.end(),
        queryCorrelationId,
        [](uint32_t val, const WaitEventInfo& info) {
          return val < info.correlationId;
        });
    if (pos != vec.begin()) {
      return *std::prev(pos);
    }
  }
  return std::nullopt;
}

void CuptiActivityProfiler::handleCudaEventActivity(
    const CUpti_ActivityCudaEventType* activity,
    ActivityLogger* logger) {
  VLOG(2) << ": CUPTI_ACTIVITY_KIND_CUDA_EVENT"
          << " corrId=" << activity->correlationId
          << " eventId=" << activity->eventId
          << " streamId=" << activity->streamId
          << " contextId=" << activity->contextId;

  // Create and log the CUDA event activity
  const ITraceActivity* linked =
      linkedActivity(activity->correlationId, cpuCorrelationMap_);
  const auto& cuda_event_activity =
      traceBuffers_->addActivityWrapper(CudaEventActivity(activity, linked));

  if (outOfRange(cuda_event_activity)) {
    return;
  }

  auto device_id = contextIdtoDeviceId(activity->contextId);
  if (static_cast<int32_t>(activity->streamId) != -1) {
    recordStream(device_id, activity->streamId, "");
  } else {
    recordDevice(device_id);
  }

  VLOG(2) << "Logging CUDA event activity device = " << device_id
          << " stream = " << activity->streamId;
  cuda_event_activity.log(*logger);
  setGpuActivityPresent(true);
}

void CuptiActivityProfiler::handleCudaSyncActivity(
    const CUpti_ActivitySynchronization* activity,
    ActivityLogger* logger) {
  VLOG(2) << ": CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
          << " type=" << syncTypeString(activity->type)
          << " corrId=" << activity->correlationId
          << " streamId=" << activity->streamId
          << " eventId=" << activity->cudaEventId
          << " contextId=" << activity->contextId;

  if (!config().activitiesCudaSyncWaitEvents() &&
      isWaitEventSync(activity->type)) {
    return;
  }

  auto device_id = contextIdtoDeviceId(activity->contextId);
  int32_t src_stream = -1;
  int32_t src_corrid = -1;
  if (isEventSync(activity->type)) {
    auto maybe_wait_event_info = getWaitEventInfo(
        activity->contextId, activity->cudaEventId, activity->correlationId);
    if (maybe_wait_event_info) {
      src_stream = maybe_wait_event_info->stream;
      src_corrid = maybe_wait_event_info->correlationId;
    }
  }

  // Marshal the logging to a functor so we can defer it if needed.
  auto log_event =
      [activity, src_stream, src_corrid, device_id, logger, this]() {
        const ITraceActivity* linked =
            linkedActivity(activity->correlationId, this->cpuCorrelationMap_);
        const auto& cuda_sync_activity =
            this->traceBuffers_->addActivityWrapper(
                CudaSyncActivity(activity, linked, src_stream, src_corrid));

        if (outOfRange(cuda_sync_activity)) {
          return;
        }

        if (static_cast<int32_t>(activity->streamId) != -1) {
          recordStream(device_id, activity->streamId, "");
        } else {
          recordDevice(device_id);
        }
        VLOG(2) << "Logging sync event device = " << device_id
                << " stream = " << activity->streamId
                << " sync type = " << syncTypeString(activity->type);
        cuda_sync_activity.log(*logger);
        setGpuActivityPresent(true);
      };

  if (isWaitEventSync(activity->type)) {
    // Stream Wait Events still need deferral so we can suppress waits on
    // streams that never show GPU work in the trace window.
    DeferredLogEntry entry;
    entry.device = device_id;
    entry.stream = activity->streamId;
    entry.logMe = log_event;

    logQueue_.push_back(entry);
  } else {
    log_event();
  }
}

void CuptiActivityProfiler::logDeferredEvents() {
  // Stream Wait Events tend to be noisy, only pass these events if
  // there was some GPU kernel/memcopy/memset observed on it in the trace
  // window.
  for (const auto& entry : logQueue_) {
    if (!seenDeviceStreams_.contains({entry.device, entry.stream})) {
      VLOG(2) << "Skipping Stream Wait Event on unseen stream = "
              << entry.stream;
    } else {
      entry.logMe();
    }
  }
}

template <class T>
inline void CuptiActivityProfiler::handleGpuActivity(
    const T* act,
    ActivityLogger* logger) {
  const ITraceActivity* linked =
      linkedActivity(act->correlationId, cpuCorrelationMap_);
  const auto& gpu_activity =
      traceBuffers_->addActivityWrapper(GpuActivity<T>(act, linked));
  GenericActivityProfiler::handleGpuActivity(gpu_activity, logger);
}

uint32_t contextIdtoDeviceId(uint32_t contextId) {
  auto it = ctxToDeviceId().find(contextId);
  return it != ctxToDeviceId().end() ? it->second : 0;
}

void CuptiActivityProfiler::handleCuptiActivity(
    const CUpti_Activity* record,
    ActivityLogger* logger) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
      // Populated in buildProcessingState().
      break;
    case CUPTI_ACTIVITY_KIND_RUNTIME:
      handleRuntimeActivity(
          reinterpret_cast<const CUpti_ActivityAPI*>(record), logger);
      break;
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      const CUpti_ActivityKernelType* kernel =
          reinterpret_cast<const CUpti_ActivityKernelType*>(record);
      // Register all kernels launches so we could correlate them with other
      // events.
      KernelRegistry::singleton()->recordKernel(
          kernel->deviceId, demangle(kernel->name), kernel->correlationId);
      handleGpuActivity(kernel, logger);
      break;
    }
    case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
      handleCudaSyncActivity(
          reinterpret_cast<const CUpti_ActivitySynchronization*>(record),
          logger);
      break;
    case CUPTI_ACTIVITY_KIND_CUDA_EVENT:
      handleCudaEventActivity(
          reinterpret_cast<const CUpti_ActivityCudaEventType*>(record), logger);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY:
      handleGpuActivity(
          reinterpret_cast<const CUpti_ActivityMemcpyType*>(record), logger);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY2:
      handleGpuActivity(
          reinterpret_cast<const CUpti_ActivityMemcpyPtoPType*>(record),
          logger);
      break;
    case CUPTI_ACTIVITY_KIND_MEMSET:
      handleGpuActivity(
          reinterpret_cast<const CUpti_ActivityMemsetType*>(record), logger);
      break;
    case CUPTI_ACTIVITY_KIND_OVERHEAD:
      handleOverheadActivity(
          reinterpret_cast<const CUpti_ActivityOverhead*>(record), logger);
      break;
    case CUPTI_ACTIVITY_KIND_DRIVER:
      handleDriverActivity(
          reinterpret_cast<const CUpti_ActivityAPI*>(record), logger);
      break;
    default:
      LOG(WARNING) << "Unexpected activity type: " << record->kind;
      ecs_.unexepected_cuda_events++;
      break;
  }
}

} // namespace KINETO_NAMESPACE
