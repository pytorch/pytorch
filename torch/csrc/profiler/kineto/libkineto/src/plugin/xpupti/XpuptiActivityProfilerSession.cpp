/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityProfilerSession.h"
#include "TraceSpan.h"
#include "XpuptiActivityApi.h"
#include "XpuptiProfilerMacros.h"

#include "Logger.h"
#include "time_since_epoch.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <ostream>
#include <ranges>

#include <fmt/format.h>
#include <pti/pti_version.h>
#include <sycl/sycl.hpp>

namespace KINETO_NAMESPACE {

namespace {
// Reads system_clock between two PTI-clock reads and keeps the tightest of a
// few tries, so both values refer to the same instant to well under 1 us.
XpuptiClockSample sampleClocks() {
  XpuptiClockSample best;
  int64_t bestSpan = std::numeric_limits<int64_t>::max();
  for (int i = 0; i < 5; ++i) {
    const auto before = static_cast<int64_t>(XpuptiActivityApi::ptiClockNs());
    const int64_t sys =
        libkineto::timeSinceEpoch(std::chrono::system_clock::now());
    const auto after = static_cast<int64_t>(XpuptiActivityApi::ptiClockNs());
    if (after - before < bestSpan) {
      bestSpan = after - before;
      best = {before + (after - before) / 2, sys};
    }
  }
  return best;
}
} // namespace

uint32_t XpuptiActivityProfilerSession::iterationCount_ = 0;
std::vector<DeviceUUIDsT> XpuptiActivityProfilerSession::deviceUUIDs_ = {};

// =========== Session Constructor ============= //
XpuptiActivityProfilerSession::XpuptiActivityProfilerSession(
    XpuptiActivityApi& xpti,
    const std::string& name,
    const libkineto::Config& config,
    const std::set<ActivityType>& activity_types)
    : xpti_(xpti),
      config_(config.clone()),
      tracedTypes_(activity_types),
      name_(name) {
  enumDeviceUUIDs();
  xpti_.enableXpuptiActivities(tracedTypes_);
  ptiClockIsMonotonicRaw_ = xpti_.ptiClockIsMonotonicRaw();
  clockAtStart_ = sampleClocks();
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
  xpti_.disablePtiActivities(tracedTypes_);
  profilerEndTs_ = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
}

void XpuptiActivityProfilerSession::toggleCollectionDynamic(const bool enable) {
  if (enable) {
    xpti_.enableXpuptiActivities(tracedTypes_);
  } else {
    xpti_.disablePtiActivities(tracedTypes_);
  }
}

void XpuptiActivityProfilerSession::processTrace(ActivityLogger& logger) {
  const XpuptiClockSample clockAtEnd = sampleClocks();
  clockRateError_ = 0.0;
  const int64_t dPti = clockAtEnd.pti - clockAtStart_.pti;
  // Under ~10 ms the samples' jitter (tens of ns) would dominate the rate.
  if (ptiClockIsMonotonicRaw_ && dPti >= 10'000'000) {
    const int64_t dSys = clockAtEnd.sys - clockAtStart_.sys;
    clockRateError_ =
        static_cast<double>(dSys - dPti) / static_cast<double>(dPti);
    if (std::abs(clockRateError_) > 500e-6) {
      LOG(WARNING) << "system_clock moved " << (dSys - dPti)
                   << " ns against CLOCK_MONOTONIC_RAW in " << dPti
                   << " ns (> 500 ppm); XPU activity times are mapped linearly";
    }
  }
  traceBuffer_.span =
      libkineto::TraceSpan(profilerStartTs_, profilerEndTs_, name_);
  traceBuffer_.span.iteration = iterationCount_++;
  auto gpuBuffer = xpti_.activityBuffers();
  if (gpuBuffer) {
    const auto stats = xpti_.processActivities(
        *gpuBuffer,
        [this, &logger](const pti_view_record_base* record) -> void {
          handlePtiActivity(record, logger);
        });
    LOG(INFO) << "Processed " << stats.activitiesCount << " GPU records ("
              << stats.buffersSize << " bytes)";
    LOGGER_OBSERVER_ADD_EVENT_COUNT(stats.activitiesCount);
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

std::unordered_map<std::string, std::string> XpuptiActivityProfilerSession::
    getMetadata() {
  const char* version = ptiVersionString();
  return {
      {"xpupti_version",
       fmt::format("\"{}\"", version != nullptr ? version : "unknown")}};
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
