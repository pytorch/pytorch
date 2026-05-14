#ifdef USE_KINETO

#include "profiler/OpenRegActivityProfilerSession.h"

#include <output_base.h> // ActivityLogger complete definition
#include <libkineto.h>
#include <c10/util/ApproximateClock.h>

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
  OpenRegTracer::instance().enableActivityTracing();
  status_ = libkineto::TraceStatus::RECORDING;
}

void OpenRegActivityProfilerSession::stop() {
  OpenRegTracer::instance().disableActivityTracing();
  endTs_ = nowUs();
  status_ = libkineto::TraceStatus::PROCESSING;
}

std::vector<std::string> OpenRegActivityProfilerSession::errors() {
  return errors_;
}

void OpenRegActivityProfilerSession::processTrace(
    libkineto::ActivityLogger& /*logger*/) {
  traceBuffer_.span = libkineto::TraceSpan(startTs_, endTs_, "openreg");
  // auto records = OpenRegTracer::instance().flush();
  // convertRecords(records, logger);
}

void OpenRegActivityProfilerSession::processTrace(
    libkineto::ActivityLogger& logger,
    libkineto::getLinkedActivityCallback getLinkedActivity,
    int64_t /*captureWindowStartTime*/,
    int64_t /*captureWindowEndTime*/) {
  // captureWindowStartTime_ = captureWindowStartTime / kNsPerUs;
  // captureWindowEndTime_ = captureWindowEndTime / kNsPerUs;
  // cpuActivity_ = getLinkedActivity;
  processTrace(logger);
}

// ---------------------------------------------------------------------------
// Trace-viewer rows
// ---------------------------------------------------------------------------

std::unique_ptr<libkineto::DeviceInfo>
OpenRegActivityProfilerSession::getDeviceInfo() {
  return std::make_unique<libkineto::DeviceInfo>(
      deviceIndex_, deviceIndex_, "OpenReg",
      "OpenReg " + std::to_string(deviceIndex_));
}

std::vector<libkineto::ResourceInfo>
OpenRegActivityProfilerSession::getResourceInfos() {
  return {};
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
  OpenRegTracer::instance().pushExternalCorrelationId(id);
}

void OpenRegActivityProfilerSession::popCorrelationId() {
  OpenRegTracer::instance().popExternalCorrelationId();
}

} // namespace openreg::profiler

#endif // USE_KINETO