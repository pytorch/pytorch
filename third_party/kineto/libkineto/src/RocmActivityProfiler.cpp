/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef HAS_ROCTRACER

#include "RocmActivityProfiler.h"
#include <fmt/format.h>
#include <string>
#include "DeviceUtil.h"

#include <rocprofiler-sdk/version.h>

#include "ActivityBuffers.h"
#include "Config.h"
#include "Logger.h"
// RocprofActivity.h must stay in this .cpp only — its corresponding _inl.h has
// inline functions referencing thread_local maps.
#include "RocprofActivity.h"
#include "ThreadUtil.h"

using namespace std::chrono;
using std::string;

namespace KINETO_NAMESPACE {

RocmActivityProfiler::RocmActivityProfiler(
    RocprofActivityApi& rocprof,
    bool cpuOnly)
    : GenericActivityProfiler(cpuOnly), roc_(rocprof) {
  if (isGpuAvailable()) {
    logGpuVersions();
  }
}

void RocmActivityProfiler::logGpuVersions() {
  uint32_t majorVersion = ROCPROFILER_VERSION_MAJOR;
  uint32_t minorVersion = ROCPROFILER_VERSION_MINOR;
  std::string rocprofVersion =
      std::to_string(majorVersion) + "." + std::to_string(minorVersion);
  int hipRuntimeVersion = 0, hipDriverVersion = 0;
  CUDA_CALL(hipRuntimeGetVersion(&hipRuntimeVersion));
  CUDA_CALL(hipDriverGetVersion(&hipDriverVersion));
  LOG(INFO) << "HIP versions. Rocprofiler-sdk: " << rocprofVersion
            << "; Runtime: " << hipRuntimeVersion
            << "; Driver: " << hipDriverVersion;

  LOGGER_OBSERVER_ADD_PERSISTENT_METADATA(
      "rocprofiler-sdk_version", rocprofVersion);
  LOGGER_OBSERVER_ADD_PERSISTENT_METADATA(
      "hip_runtime_version", std::to_string(hipRuntimeVersion));
  LOGGER_OBSERVER_ADD_PERSISTENT_METADATA(
      "hip_driver_version", std::to_string(hipDriverVersion));
  addVersionMetadata("rocprofiler-sdk_version", rocprofVersion);
  addVersionMetadata("hip_runtime_version", std::to_string(hipRuntimeVersion));
  addVersionMetadata("hip_driver_version", std::to_string(hipDriverVersion));
}

void RocmActivityProfiler::setMaxGpuBufferSize(int64_t size) {
  roc_.setMaxBufferSize(size);
}

void RocmActivityProfiler::enableGpuTracing() {
  roc_.setMaxEvents(config().maxEvents());
  roc_.enableActivities(derivedConfig_->profileActivityTypes());
}

void RocmActivityProfiler::disableGpuTracing() {
  roc_.disableActivities(derivedConfig_->profileActivityTypes());
}

void RocmActivityProfiler::clearGpuActivities() {
  roc_.clearActivities();
}

bool RocmActivityProfiler::isGpuCollectionStopped() const {
  return roc_.stopCollection;
}

void RocmActivityProfiler::synchronizeGpuDevice() {
  CUDA_CALL(hipDeviceSynchronize());
  roc_.flushActivities();
}

void RocmActivityProfiler::pushCorrelationIdImpl(
    uint64_t id,
    CorrelationFlowType type) {
  RocprofActivityApi::CorrelationFlowType rocprofType =
      (type == CorrelationFlowType::User)
      ? RocprofActivityApi::CorrelationFlowType::User
      : RocprofActivityApi::CorrelationFlowType::Default;
  RocprofActivityApi::pushCorrelationID(id, rocprofType);
}

void RocmActivityProfiler::popCorrelationIdImpl(CorrelationFlowType type) {
  RocprofActivityApi::CorrelationFlowType rocprofType =
      (type == CorrelationFlowType::User)
      ? RocprofActivityApi::CorrelationFlowType::User
      : RocprofActivityApi::CorrelationFlowType::Default;
  RocprofActivityApi::popCorrelationID(rocprofType);
}

void RocmActivityProfiler::onResetTraceData() {
  roc_.teardownContext();
}

void RocmActivityProfiler::onFinalizeTrace(
    [[maybe_unused]] const Config& config,
    [[maybe_unused]] ActivityLogger& logger) {
  // No additional overhead info for ROCm currently
}

void RocmActivityProfiler::processGpuActivities(ActivityLogger& logger) {
  VLOG(0) << "Retrieving GPU activity buffers";
  const int count = roc_.processActivities(
      std::bind(
          &RocmActivityProfiler::handleRocprofActivity,
          this,
          std::placeholders::_1,
          &logger),
      std::bind(
          &RocmActivityProfiler::handleCorrelationActivity,
          this,
          std::placeholders::_1,
          std::placeholders::_2,
          std::placeholders::_3));
  LOG(INFO) << "Processed " << count << " GPU records";
  LOGGER_OBSERVER_ADD_EVENT_COUNT(count);
}

inline void RocmActivityProfiler::handleCorrelationActivity(
    uint64_t correlationId,
    uint64_t externalId,
    RocLogger::CorrelationDomain externalKind) {
  if (externalKind == RocLogger::CorrelationDomain::Domain0) {
    cpuCorrelationMap_[correlationId] = externalId;
  } else if (externalKind == RocLogger::CorrelationDomain::Domain1) {
    userCorrelationMap_[correlationId] = externalId;
  } else {
    LOG(WARNING)
        << "Invalid RocLogger::CorrelationDomain sent to handleCorrelationActivity";
    ecs_.invalid_external_correlation_events++;
  }
}

template <class T>
void RocmActivityProfiler::handleRuntimeActivity(
    const T* activity,
    ActivityLogger* logger) {
  int32_t tid = activity->tid;
  const auto& it = resourceInfo_.find({processId(), tid});
  if (it != resourceInfo_.end()) {
    tid = it->second.id;
  }
  const ITraceActivity* linked =
      linkedActivity(activity->id, cpuCorrelationMap_);
  const auto& runtime_activity =
      traceBuffers_->addActivityWrapper(RuntimeActivity<T>(activity, linked));
  checkTimestampOrder(&runtime_activity);
  if (outOfRange(runtime_activity)) {
    return;
  }
  runtime_activity.log(*logger);
  setGpuActivityPresent(true);
}

inline void RocmActivityProfiler::handleGpuActivity(
    const rocprofAsyncRow* act,
    ActivityLogger* logger) {
  const ITraceActivity* linked = linkedActivity(act->id, cpuCorrelationMap_);
  const auto& gpu_activity =
      traceBuffers_->addActivityWrapper(GpuActivity(act, linked));
  GenericActivityProfiler::handleGpuActivity(gpu_activity, logger);
}

void RocmActivityProfiler::handleRocprofActivity(
    const rocprofBase* record,
    ActivityLogger* logger) {
  switch (record->type) {
    case ROCTRACER_ACTIVITY_DEFAULT:
      handleRuntimeActivity(
          reinterpret_cast<const rocprofRow*>(record), logger);
      break;
    case ROCTRACER_ACTIVITY_KERNEL:
      handleRuntimeActivity(
          reinterpret_cast<const rocprofKernelRow*>(record), logger);
      break;
    case ROCTRACER_ACTIVITY_COPY:
      handleRuntimeActivity(
          reinterpret_cast<const rocprofCopyRow*>(record), logger);
      break;
    case ROCTRACER_ACTIVITY_MALLOC:
      handleRuntimeActivity(
          reinterpret_cast<const rocprofMallocRow*>(record), logger);
      break;
    case ROCTRACER_ACTIVITY_ASYNC:
      handleGpuActivity(
          reinterpret_cast<const rocprofAsyncRow*>(record), logger);
      break;
    case ROCTRACER_ACTIVITY_NONE:
    default:
      LOG(WARNING) << "Unexpected activity type: " << record->type;
      ecs_.unexepected_cuda_events++;
      break;
  }
}

} // namespace KINETO_NAMESPACE

#endif // HAS_ROCTRACER
