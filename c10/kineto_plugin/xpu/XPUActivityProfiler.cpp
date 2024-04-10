#include "XPUActivityProfiler.h"
#include "XPUActivityApi.h"

#include <chrono>

namespace c10::kineto_plugin::xpu {

uint32_t XPUActivityProfilerSession::iterationCount_ = 0;

// =========== Session Constructor ============= //
XPUActivityProfilerSession::XPUActivityProfilerSession(
    XPUActivityApi& xpti,
    const libkineto::Config& config,
    const std::set<act_t>& activity_types)
  : xpti_(xpti),
    config_(config.clone()),
    activity_types_(activity_types) {
  xpti_.setMaxBufferSize(config_->activitiesMaxGpuBufferSize());
  xpti_.enablePtiActivities(activity_types_);
}

XPUActivityProfilerSession::~XPUActivityProfilerSession() {
  xpti_.clearActivities();
}


// =========== Session Public Methods ============= //
void XPUActivityProfilerSession::start() {
  profilerStartTs_ = libkineto::timeSinceEpoch(
		  std::chrono::high_resolution_clock::now());
}

void XPUActivityProfilerSession::stop() {
  xpti_.disablePtiActivities(activity_types_);
  profilerEndTs_ = libkineto::timeSinceEpoch(
		  std::chrono::high_resolution_clock::now());
}

void XPUActivityProfilerSession::processTrace(logger_t& logger) {
  traceBuffer_.span = libkineto::TraceSpan(
      profilerStartTs_, profilerEndTs_, "__xpu_profiler__");
  traceBuffer_.span.iteration = iterationCount_++;
  auto gpuBuffer = xpti_.activityBuffers();
  if (gpuBuffer) {
    xpti_.processActivities(
	*gpuBuffer,
        std::bind(
          &XPUActivityProfilerSession::handlePtiActivity,
          this,
          std::placeholders::_1,
          &logger));
  }
}

void XPUActivityProfilerSession::processTrace(
    logger_t& logger,
    libkineto::getLinkedActivityCallback get_linked_activity,
    int64_t captureWindowStartTime,
    int64_t captureWindowEndTime) {
  captureWindowStartTime_ = captureWindowStartTime;
  captureWindowEndTime_ = captureWindowEndTime;
  cpuActivity_ = get_linked_activity;
  processTrace(logger);
}

std::unique_ptr<libkineto::DeviceInfo>
XPUActivityProfilerSession::getDeviceInfo() {
  return {};
}

std::vector<libkineto::ResourceInfo>
XPUActivityProfilerSession::getResourceInfos() {
  return {};
}

std::unique_ptr<libkineto::CpuTraceBuffer>
XPUActivityProfilerSession::getTraceBuffer() {
  return std::make_unique<libkineto::CpuTraceBuffer>(std::move(traceBuffer_));
}

void XPUActivityProfilerSession::pushCorrelationId(uint64_t id) {
  xpti_.pushCorrelationID(id, XPUActivityApi::CorrelationFlowType::Default);
}

void XPUActivityProfilerSession::popCorrelationId() {
  xpti_.popCorrelationID(XPUActivityApi::CorrelationFlowType::Default);
}

void XPUActivityProfilerSession::pushUserCorrelationId(uint64_t id) {
  xpti_.pushCorrelationID(id, XPUActivityApi::CorrelationFlowType::User);
}

void XPUActivityProfilerSession::popUserCorrelationId() {
  xpti_.popCorrelationID(XPUActivityApi::CorrelationFlowType::User);
}

// =========== ActivityProfiler Public Methods ============= //
const std::set<act_t> kXpuTypes {
  act_t::GPU_MEMCPY,
  act_t::GPU_MEMSET,
  act_t::CONCURRENT_KERNEL,
  act_t::XPU_RUNTIME,
  // act_t::EXTERNAL_CORRELATION,
  // act_t::Overhead,
};

const std::string& XPUActivityProfiler::name() const {
  return name_;
}

const std::set<act_t>& XPUActivityProfiler::availableActivities() const {
  return kXpuTypes;
}

std::unique_ptr<libkineto::IActivityProfilerSession>
XPUActivityProfiler::configure(
    const std::set<act_t>& activity_types,
    const libkineto::Config& config) {
  return std::make_unique<XPUActivityProfilerSession>(
      XPUActivityApi::singleton(), config, activity_types);
}

std::unique_ptr<libkineto::IActivityProfilerSession>
XPUActivityProfiler::configure(
    int64_t ts_ms,
    int64_t duration_ms,
    const std::set<act_t>& activity_types,
    const libkineto::Config& config) {
  AsyncProfileStartTime_ = ts_ms;
  AsyncProfileEndTime_ = ts_ms + duration_ms;
  return configure(activity_types, config);
}

}
