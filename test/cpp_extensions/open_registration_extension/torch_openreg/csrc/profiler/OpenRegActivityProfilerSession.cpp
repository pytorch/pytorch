#ifdef USE_KINETO

#include "profiler/OpenRegActivityProfilerSession.h"

#include <output_base.h>   // ActivityLogger complete definition
#include <libkineto.h>
#include <c10/util/ApproximateClock.h>

#include <algorithm>
#include <string>

namespace openreg::profiler {

namespace {

constexpr int64_t kNsPerUs = 1000;

int64_t nowUs() {
  return c10::getTime() / kNsPerUs;
}

} // namespace

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void OpenRegActivityProfilerSession::start() {
  startTs_ = nowUs();
  OpenRegTracer::instance().enable();
  status_ = libkineto::TraceStatus::RECORDING;
}

void OpenRegActivityProfilerSession::stop() {
  OpenRegTracer::instance().disable();
  endTs_ = nowUs();
  status_ = libkineto::TraceStatus::PROCESSING;
}

std::vector<std::string> OpenRegActivityProfilerSession::errors() {
  return errors_;
}

libkineto::ActivityType OpenRegActivityProfilerSession::toActivityType(
    ActivityKind kind) {
  switch (kind) {
    case ActivityKind::KERNEL:
      return libkineto::ActivityType::CONCURRENT_KERNEL;
    case ActivityKind::MEMCPY:
      return libkineto::ActivityType::GPU_MEMCPY;
    case ActivityKind::MEMSET:
      return libkineto::ActivityType::GPU_MEMSET;
    case ActivityKind::RUNTIME:
      return libkineto::ActivityType::PRIVATEUSE1_RUNTIME;
    case ActivityKind::DRIVER:
      return libkineto::ActivityType::PRIVATEUSE1_DRIVER;
  }
  return libkineto::ActivityType::CONCURRENT_KERNEL; // unreachable
}

// ---------------------------------------------------------------------------
// Core conversion loop
// ---------------------------------------------------------------------------

void OpenRegActivityProfilerSession::convertRecords(
    const std::vector<TraceRecord>& records,
    libkineto::ActivityLogger& logger) {
  const bool filter =
      (captureWindowStartTime_ != 0 || captureWindowEndTime_ != 0);

  for (const auto& rec : records) {
    const int64_t startUs = rec.start / kNsPerUs;
    const int64_t endUs   = rec.end   / kNsPerUs;

    if (filter &&
        (endUs < captureWindowStartTime_ || startUs > captureWindowEndTime_)) {
      continue;
    }

    // Track every (device, stream) pair for getResourceInfos().
    const auto resPair = std::make_pair(rec.device_id, rec.stream_id);
    if (std::find(resources_.begin(), resources_.end(), resPair) ==
        resources_.end()) {
      resources_.push_back(resPair);
    }

    traceBuffer_.emplace_activity(traceBuffer_.span, toActivityType(rec.kind), rec.name);
    auto& act = libkineto::CpuTraceBuffer::toRef(traceBuffer_.activities.back());
    act.startTime = startUs;
    act.endTime   = endUs;
    act.device    = rec.device_id;
    act.resource  = rec.stream_id;
    act.id        = static_cast<int32_t>(rec.correlation_id);

    // flow links this device event back to the CPU-side op that launched it.
    act.flow.id    = static_cast<uint32_t>(rec.correlation_id);
    act.flow.type  = libkineto::kLinkAsyncCpuGpu;
    act.flow.start = 0; // device side = end of the async CPU→device flow

    if (cpuActivity_) {
      act.linked =
          cpuActivity_(static_cast<int32_t>(rec.correlation_id));
    }

    logger.handleActivity(act);
  }
}

void OpenRegActivityProfilerSession::processTrace(
    libkineto::ActivityLogger& logger) {
  traceBuffer_.span = libkineto::TraceSpan(startTs_, endTs_, "openreg");
  auto records = OpenRegTracer::instance().flush();
  convertRecords(records, logger);
}

void OpenRegActivityProfilerSession::processTrace(
    libkineto::ActivityLogger& logger,
    libkineto::getLinkedActivityCallback getLinkedActivity,
    int64_t captureWindowStartTime,
    int64_t captureWindowEndTime) {
  captureWindowStartTime_ = captureWindowStartTime / kNsPerUs;
  captureWindowEndTime_   = captureWindowEndTime / kNsPerUs;
  cpuActivity_            = getLinkedActivity;
  processTrace(logger);
}

// ---------------------------------------------------------------------------
// Trace-viewer rows
// ---------------------------------------------------------------------------

std::unique_ptr<libkineto::DeviceInfo>
OpenRegActivityProfilerSession::getDeviceInfo() {
  return std::make_unique<libkineto::DeviceInfo>(
      /*id=*/0, /*sortIndex=*/0, /*name=*/"OpenReg", /*label=*/"OpenReg 0");
}

std::vector<libkineto::ResourceInfo>
OpenRegActivityProfilerSession::getResourceInfos() {
  std::vector<libkineto::ResourceInfo> result;
  result.reserve(resources_.size());
  for (const auto& [deviceId, streamId] : resources_) {
    result.emplace_back(
        deviceId,
        streamId,
        /*sortIndex=*/static_cast<int64_t>(streamId),
        "stream " + std::to_string(streamId));
  }
  return result;
}

// ---------------------------------------------------------------------------
// Buffer handoff
// ---------------------------------------------------------------------------

std::unique_ptr<libkineto::CpuTraceBuffer>
OpenRegActivityProfilerSession::getTraceBuffer() {
  return std::make_unique<libkineto::CpuTraceBuffer>(std::move(traceBuffer_));
}

// ---------------------------------------------------------------------------
// Correlation forwarding
// ---------------------------------------------------------------------------

void OpenRegActivityProfilerSession::pushCorrelationId(uint64_t id) {
  OpenRegTracer::instance().pushCorrelation(id);
}

void OpenRegActivityProfilerSession::popCorrelationId() {
  OpenRegTracer::instance().popCorrelation();
}

} // namespace openreg::profiler

#endif // USE_KINETO