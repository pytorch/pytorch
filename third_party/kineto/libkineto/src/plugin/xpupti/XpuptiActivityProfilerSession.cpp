/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityProfilerSession.h"
#include "XpuptiActivityApi.h"

#include "time_since_epoch.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <iterator>

namespace KINETO_NAMESPACE {

uint32_t XpuptiActivityProfilerSession::iterationCount_ = 0;
std::vector<DeviceUUIDsT> XpuptiActivityProfilerSession::deviceUUIDs_ = {};

bool XpuptiActivityProfilerSession::startsFlow(ActivityType activityType) {
  // Only host runtime records start the CPU->GPU flow. The runtime view is
  // already filtered to work-submitting APIs via
  // ptiViewEnableRuntimeApiClass(PTI_API_CLASS_GPU_OPERATION_CORE). Driver
  // (XPU_DRIVER) records share the same correlation id as the runtime record
  // and must not also start a flow, or the trace gets a duplicate flow start.
  return activityType == ActivityType::XPU_RUNTIME;
}

// =========== Session Constructor ============= //
XpuptiActivityProfilerSession::XpuptiActivityProfilerSession(
    XpuptiActivityApi& xpti,
    const std::string& name,
    const libkineto::Config& config,
    const std::set<ActivityType>& activity_types)
    : xpti_(xpti),
      config_(config.clone()),
      activity_types_(activity_types),
      name_(name) {
  enumDeviceUUIDs();
  xpti_.enableXpuptiActivities(activity_types_);
}

XpuptiActivityProfilerSession::~XpuptiActivityProfilerSession() {
  xpti_.clearActivities();
}

// =========== Session Public Methods ============= //
void XpuptiActivityProfilerSession::start() {
  profilerStartTs_ =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
}

void XpuptiActivityProfilerSession::stop() {
  xpti_.disablePtiActivities(activity_types_);
  profilerEndTs_ = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
}

void XpuptiActivityProfilerSession::toggleCollectionDynamic(const bool enable) {
  if (enable) {
    xpti_.enableXpuptiActivities(activity_types_);
  } else {
    xpti_.disablePtiActivities(activity_types_);
  }
}

void XpuptiActivityProfilerSession::processTrace(ActivityLogger& logger) {
  traceBuffer_.span =
      libkineto::TraceSpan(profilerStartTs_, profilerEndTs_, name_);
  traceBuffer_.span.iteration = iterationCount_++;
  auto gpuBuffer = xpti_.activityBuffers();
  if (gpuBuffer) {
    xpti_.processActivities(
        *gpuBuffer,
        [this, &logger](const pti_view_record_base* record) -> void {
          handlePtiActivity(record, logger);
        });
  }
  for (auto& kv : userAnnotationsByStream_) {
    kv.second->log(logger);
  }
  userAnnotationsByStream_.clear();
}

void XpuptiActivityProfilerSession::processTrace(
    ActivityLogger& logger,
    libkineto::getLinkedActivityCallback get_linked_activity,
    int64_t captureWindowStartTime,
    int64_t captureWindowEndTime) {
  captureWindowStartTime_ = captureWindowStartTime;
  captureWindowEndTime_ = captureWindowEndTime;
  cpuActivity_ = get_linked_activity;
  processTrace(logger);
}

std::unique_ptr<libkineto::CpuTraceBuffer> XpuptiActivityProfilerSession::
    getTraceBuffer() {
  return std::make_unique<libkineto::CpuTraceBuffer>(std::move(traceBuffer_));
}

std::vector<libkineto::ResourceInfo> XpuptiActivityProfilerSession::
    getResourceInfos() {
  std::vector<libkineto::ResourceInfo> result;
  for (const auto& [device_id, sycl_queue_id] : resourceInfo_) {
    result.push_back(
        {.id = sycl_queue_id,
         .sortIndex = sycl_queue_id,
         .deviceId = device_id,
         .name = fmt::format("Stream {}", sycl_queue_id)});
  }
  resourceInfo_.clear();
  return result;
}

void XpuptiActivityProfilerSession::pushCorrelationId(uint64_t id) {
  xpti_.pushCorrelationID(id, XpuptiActivityApi::CorrelationFlowType::Default);
}

void XpuptiActivityProfilerSession::popCorrelationId() {
  xpti_.popCorrelationID(XpuptiActivityApi::CorrelationFlowType::Default);
}

void XpuptiActivityProfilerSession::pushUserCorrelationId(uint64_t id) {
  xpti_.pushCorrelationID(id, XpuptiActivityApi::CorrelationFlowType::User);
}

void XpuptiActivityProfilerSession::popUserCorrelationId() {
  xpti_.popCorrelationID(XpuptiActivityApi::CorrelationFlowType::User);
}

void XpuptiActivityProfilerSession::enumDeviceUUIDs() {
  if (!deviceUUIDs_.empty()) {
    return;
  }
  auto platform_list = sycl::platform::get_platforms();
  // Enumerated GPU devices from the specific platform.
  for (const auto& platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        if (device.has(sycl::aspect::ext_intel_device_info_uuid)) {
          deviceUUIDs_.push_back(
              device.get_info<sycl::ext::intel::info::device::uuid>());
        } else {
          std::cerr
              << "Warnings: UUID is not supported for this XPU device. The device index of records will be 0."
              << std::endl;
          deviceUUIDs_.push_back(DeviceUUIDsT{});
        }
      }
    }
  }
}

DeviceIndex_t XpuptiActivityProfilerSession::getDeviceIdxFromUUID(
    const uint8_t deviceUUID[16]) {
  auto it = std::ranges::find_if(
      deviceUUIDs_, [deviceUUID](const DeviceUUIDsT& deviceUUIDinVec) {
        return std::equal(
            deviceUUIDinVec.begin(),
            deviceUUIDinVec.end(),
            deviceUUID,
            deviceUUID + 16);
      });
  if (it == deviceUUIDs_.end()) {
    std::cerr
        << "Warnings: Can't find the legal XPU device from the given UUID."
        << std::endl;
    return static_cast<DeviceIndex_t>(0);
  }
  return static_cast<DeviceIndex_t>(std::distance(deviceUUIDs_.begin(), it));
}

} // namespace KINETO_NAMESPACE
