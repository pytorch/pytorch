#include "XPUActivityProfiler.h"
#include "XPUActivityApi.h"
#include "XPUActivity.h"

namespace c10::xpu {

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
void XPUActivityProfilerSession::start() {}

void XPUActivityProfilerSession::stop() {
  xpti_.disablePtiActivities(activity_types_);
}

void XPUActivityProfilerSession::processTrace(logger_t& logger) {
  auto gpuBuffers = xpti_.activityBuffers();
  if (gpuBuffers) {
    xpti_.processActivities(
        *gpuBuffers,
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
  // return additional cpu buffer contained activities,
  // but all current XPU acts are gpu's
  return std::make_unique<libkineto::CpuTraceBuffer>();
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

// =========== Session Private Methods ============= //
void XPUActivityProfilerSession::checkTimestampOrder(const itrace_t* act1) {
  const auto& it = correlatedPtiActivities_.find(act1->correlationId());
  if (it == correlatedPtiActivities_.end()) {
    correlatedPtiActivities_.insert({act1->correlationId(), act1});
    return;
  }

  const itrace_t* act2 = it->second;
  if (act2->type() == act_t::XPU_RUNTIME) {
    std::swap(act1, act2);
  }
  if (act1->timestamp() > act2->timestamp()) {
    std::string err_msg;
    err_msg += "GPU op timestamp (" + std::to_string(act2->timestamp());
    err_msg += ") < runtime timestamp (" + std::to_string(act1->timestamp());
    err_msg += ") by " + std::to_string(act1->timestamp() - act2->timestamp());
    err_msg += "us Name: " + act2->name();
    err_msg += " Device: " + std::to_string(act2->deviceId());
    err_msg += " Queue: " + std::to_string(act2->resourceId());
    errors_.push_back(err_msg);
  }
}

inline bool XPUActivityProfilerSession::outOfRange(const itrace_t& act) {
  bool out_of_range = act.timestamp() < captureWindowStartTime_ ||
      (act.timestamp() + act.duration()) > captureWindowEndTime_;
  if (out_of_range) {
    std::string err_msg;
    err_msg += "TraceActivity outside of profiling window: " + act.name();
    err_msg += " (" + std::to_string(act.timestamp());
    err_msg += " < " + std::to_string(captureWindowStartTime_);
    err_msg += " or " + std::to_string(act.timestamp() + act.duration());
    err_msg += " > " + std::to_string(captureWindowEndTime_);
    errors_.push_back(err_msg);
  }
  return out_of_range;
}

const itrace_t* XPUActivityProfilerSession::linkedActivity(
    int32_t correlationId,
    const std::unordered_map<int64_t, int64_t>& correlationMap) {
  const auto& it = correlationMap.find(correlationId);
  if (it != correlationMap.end()) {
    return cpuActivity_(it->second);
  }
  return nullptr;
}

inline void XPUActivityProfilerSession::handleCorrelationActivity(
    const pti_view_record_external_correlation* correlation) {
  if (correlation->_external_kind == PTI_VIEW_EXTERNAL_KIND_CUSTOM_0) {
    cpuCorrelationMap_[correlation->_correlation_id] =
        correlation->_external_id;
  } else if (correlation->_external_kind == PTI_VIEW_EXTERNAL_KIND_CUSTOM_1) {
    userCorrelationMap_[correlation->_correlation_id] =
        correlation->_external_id;
  } else {
    errors_.push_back(
        "Invalid PTI External Correaltion activity sent to handlePtiActivity");
  }
}

void XPUActivityProfilerSession::handleRuntimeActivity(
    const pti_view_record_sycl_runtime* activity,
    logger_t* logger) {
  int32_t tid = activity->_thread_id;
  const itrace_t* linked =
      linkedActivity(activity->_correlation_id, cpuCorrelationMap_);
  const auto& runtime_activity = RuntimeActivity(activity, linked, tid);
  checkTimestampOrder(&runtime_activity);
  if (outOfRange(runtime_activity)) {
    return;
  }
  runtime_activity.log(*logger);
}

inline void XPUActivityProfilerSession::handleGpuActivity(
    const itrace_t& act,
    logger_t* logger) {
  if (outOfRange(act)) {
    return;
  }
  checkTimestampOrder(&act);
  act.log(*logger);
}

template <class T>
inline void XPUActivityProfilerSession::handleGpuActivity(
    const T* act,
    logger_t* logger) {
  const itrace_t* linked =
      linkedActivity(act->_correlation_id, cpuCorrelationMap_);
  const auto& gpu_activity = GpuActivity<T>(act, linked);
  handleGpuActivity(gpu_activity, logger);
}

void XPUActivityProfilerSession::handleOverheadActivity(
    const pti_view_record_overhead* activity,
    logger_t* logger) {
  const auto& overhead_activity = OverheadActivity(activity, nullptr);

  if (outOfRange(overhead_activity)) {
    return;
  }
  overhead_activity.log(*logger);
}

void XPUActivityProfilerSession::handlePtiActivity(
    const pti_view_record_base* record,
    logger_t* logger) {
  switch (record->_view_kind) {
    case PTI_VIEW_EXTERNAL_CORRELATION:
      handleCorrelationActivity(
          reinterpret_cast<const pti_view_record_external_correlation*>(
              record));
      break;
    case PTI_VIEW_SYCL_RUNTIME_CALLS:
      handleRuntimeActivity(
          reinterpret_cast<const pti_view_record_sycl_runtime*>(record),
          logger);
      break;
    case PTI_VIEW_DEVICE_GPU_KERNEL:
      handleGpuActivity(
          reinterpret_cast<const pti_view_record_kernel*>(record), logger);
      break;
    case PTI_VIEW_DEVICE_GPU_MEM_COPY:
      handleGpuActivity(
          reinterpret_cast<const pti_view_record_memory_copy*>(record), logger);
      break;
    case PTI_VIEW_DEVICE_GPU_MEM_FILL:
      handleGpuActivity(
          reinterpret_cast<const pti_view_record_memory_fill*>(record), logger);
      break;
    case PTI_VIEW_COLLECTION_OVERHEAD:
      handleOverheadActivity(
          reinterpret_cast<const pti_view_record_overhead*>(record), logger);
      break;
    default:
      errors_.push_back("Unexpected activity type: " + std::to_string(record->_view_kind));
      break;
  }
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
  profileStartTime_ = ts_ms;
  profileEndTime_ = ts_ms + duration_ms;
  return configure(activity_types, config);
}

}
