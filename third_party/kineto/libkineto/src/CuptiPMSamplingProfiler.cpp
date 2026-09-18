/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiPMSamplingProfiler.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <string>
#include <utility>

#include "ActivityType.h"
#include "CuptiTimestamp.h"
#include "Logger.h"
#include "libkineto.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {
namespace {

const std::string kProfilerName{"CUPTI PM Sampling"};
const std::set<libkineto::ActivityType> kSupportedActivities{
    libkineto::ActivityType::HARDWARE_COUNTERS};
// TODO: This is a temporary constant -- we need to tune the interval by
// hardware type to prevent hardware buffer overflow and sample loss.
constexpr std::chrono::milliseconds kSamplingInterval{1};

} // namespace

CuptiPMSamplingSession::CuptiPMSamplingSession(
    const CuptiPMSamplingConfig& config)
    : CuptiPMSamplingSession(
          std::make_unique<CuptiPMSamplingController>(config)) {}

CuptiPMSamplingSession::CuptiPMSamplingSession(
    std::unique_ptr<ICuptiPMSamplingController> controller)
    : controller_(std::move(controller)) {}

CuptiPMSamplingSession::~CuptiPMSamplingSession() = default;

bool CuptiPMSamplingSession::prepare() {
  // PM sampling and activity profiling share the same timestamp callbacks.
  // This can only run after activity profiling enables the callbacks in
  // enableGpuTracing
  if (!isCuptiTimestampSourceReady()) {
    LOG(WARNING) << "CUPTI PM sampling requires timestamp setup from the "
                    "CUPTI Activity profiler";
    return false;
  }

  // prepare() enables CUPTI PM sampling and acquires exclusive access during
  // the parent profiler's warmup phase. Collection itself begins in start().
  if (!controller_->prepare()) {
    LOG(WARNING) << "CUPTI PM sampling failed to prepare CUDA device "
                 << controller_->deviceId();
    return false;
  }
  return true;
}

void CuptiPMSamplingSession::start() {
  controller_->start();
}

void CuptiPMSamplingSession::stop() {
  if (controller_->stop()) {
    traceBuffer_ = buildTraceBuffer();
  }
}

std::vector<std::string> CuptiPMSamplingSession::errors() {
  // GenericActivityProfiler does not consume child profiler status or errors.
  // The controller logs failures directly for visibility.
  return {};
}

void CuptiPMSamplingSession::processTrace(libkineto::ActivityLogger& logger) {
  if (!traceBuffer_) {
    return;
  }
  for (const auto& activity : traceBuffer_->activities) {
    logger.handleActivity(libkineto::CpuTraceBuffer::toRef(activity));
  }
}

void CuptiPMSamplingSession::processTrace(
    libkineto::ActivityLogger& logger,
    libkineto::getLinkedActivityCallback /*getLinkedActivity*/,
    int64_t startTime,
    int64_t endTime) {
  if (!traceBuffer_) {
    return;
  }

  // Drop PM samples that are not fully contained in [startTime, endTime].
  auto& activities = traceBuffer_->activities;
  activities.erase(
      std::remove_if(
          activities.begin(),
          activities.end(),
          [startTime, endTime](const auto& activity) {
            return activity->startTime < startTime ||
                activity->endTime > endTime;
          }),
      activities.end());

  if (activities.empty()) {
    traceBuffer_->span.startTime = startTime;
    traceBuffer_->span.endTime = endTime;
  } else {
    traceBuffer_->span.startTime = activities.front()->startTime;
    traceBuffer_->span.endTime = activities.front()->endTime;
    for (size_t i = 1; i < activities.size(); ++i) {
      traceBuffer_->span.startTime =
          std::min(traceBuffer_->span.startTime, activities[i]->startTime);
      traceBuffer_->span.endTime =
          std::max(traceBuffer_->span.endTime, activities[i]->endTime);
    }
  }

  processTrace(logger);
}

std::unique_ptr<libkineto::DeviceInfo> CuptiPMSamplingSession::getDeviceInfo() {
  return nullptr;
}

std::vector<libkineto::ResourceInfo> CuptiPMSamplingSession::
    getResourceInfos() {
  return {};
}

std::unique_ptr<libkineto::CpuTraceBuffer> CuptiPMSamplingSession::
    getTraceBuffer() {
  return std::move(traceBuffer_);
}

std::unique_ptr<libkineto::CpuTraceBuffer> CuptiPMSamplingSession::
    buildTraceBuffer() {
  auto buffer = std::make_unique<libkineto::CpuTraceBuffer>();
  buffer->span = libkineto::TraceSpan{0, 0, kProfilerName};

  const auto samples = controller_->takeSamples();
  const auto& metricNames = controller_->metricNames();
  for (const auto& sample : samples) {
    const auto start = convertCuptiTimestamp(sample.rawStartTimestamp);
    const auto end = convertCuptiTimestamp(sample.rawEndTimestamp);

    if (buffer->activities.empty()) {
      buffer->span.startTime = start;
      buffer->span.endTime = end;
    } else {
      buffer->span.startTime = std::min(buffer->span.startTime, start);
      buffer->span.endTime = std::max(buffer->span.endTime, end);
    }

    buffer->emplace_activity(
        buffer->span,
        libkineto::ActivityType::HARDWARE_COUNTERS,
        kProfilerName);
    auto& activity = *buffer->activities.back();
    activity.startTime = start;
    activity.endTime = end;
    activity.device = controller_->deviceId();
    for (size_t i = 0; i < metricNames.size(); ++i) {
      activity.addCounterValue(metricNames[i], sample.values[i]);
    }
  }

  return buffer;
}

const std::string& CuptiPMSamplingProfiler::name() const {
  return kProfilerName;
}

const std::set<libkineto::ActivityType>& CuptiPMSamplingProfiler::
    availableActivities() const {
  return kSupportedActivities;
}

std::unique_ptr<libkineto::IActivityProfilerSession> CuptiPMSamplingProfiler::
    configure(
        const std::set<libkineto::ActivityType>& /*activityTypes*/,
        const libkineto::Config& config) {
  const auto& metricNames = config.performanceMetricNames();
  if (metricNames.empty()) {
    return nullptr;
  }

  const auto deviceId = config.performanceMetricsDeviceId();
  if (deviceId < 0) {
    LOG(WARNING) << "CUPTI PM sampling requires a nonnegative "
                    "PERFORMANCE_METRICS_DEVICE_ID";
    return nullptr;
  }

  const CuptiPMSamplingConfig pmConfig{
      deviceId, metricNames, kSamplingInterval};
  auto session = std::make_unique<CuptiPMSamplingSession>(pmConfig);
  if (!session->prepare()) {
    return nullptr;
  }
  return session;
}

std::unique_ptr<libkineto::IActivityProfilerSession> CuptiPMSamplingProfiler::
    configure(
        int64_t /*startTimeMs*/,
        int64_t /*durationMs*/,
        const std::set<libkineto::ActivityType>& activityTypes,
        const libkineto::Config& config) {
  // GenericActivityProfiler calls this overload for both synchronous and
  // asynchronous traces. PM sampling does not schedule itself; the parent
  // drives collection through start() and stop(), so the timing arguments are
  // intentionally unused.
  return configure(activityTypes, config);
}

} // namespace KINETO_NAMESPACE
