#include "XPUActivityProfiler.h"

#include <c10/xpu/XPUFunctions.h>

#include <string>

namespace c10::kineto_plugin::xpu {

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

template<class ze_handle_type>
inline std::string handleToHexString(ze_handle_type handle) {
  return fmt::format("0x{:016x}", reinterpret_cast<uintptr_t>(handle));
}

// FIXME: Deprecate this method while activity._sycl_queue_id got correct IDs from PTI
inline int64_t XPUActivityProfilerSession::getMappedQueueId(uint64_t sycl_queue_id) {
  auto it = std::find(sycl_queue_pool_.begin(), sycl_queue_pool_.end(), sycl_queue_id);
  if (it != sycl_queue_pool_.end()) {
    return std::distance(sycl_queue_pool_.begin(), it);
  }
  sycl_queue_pool_.push_back(sycl_queue_id);
  return sycl_queue_pool_.size() - 1;
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
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const itrace_t* linked =
      linkedActivity(activity->_correlation_id, cpuCorrelationMap_);
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      act_t::XPU_RUNTIME,
      std::string(activity->_name));
  auto& runtime_activity = traceBuffer_.activities.back();
  runtime_activity->startTime = activity->_start_timestamp;
  runtime_activity->endTime = activity->_end_timestamp;
  runtime_activity->id = activity->_correlation_id;
  runtime_activity->device = activity->_process_id;
  runtime_activity->resource = activity->_thread_id;
  runtime_activity->threadId = activity->_thread_id;
  runtime_activity->flow.id = activity->_correlation_id;
  runtime_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  runtime_activity->flow.start = (runtime_activity->name() == "piEnqueueKernelLaunch" ? 1 : 0);
  runtime_activity->linked = linked;
  runtime_activity->addMetadata("correlation", activity->_correlation_id);
  
  checkTimestampOrder(&*runtime_activity);
  if (outOfRange(*runtime_activity)) {
    traceBuffer_.activities.pop_back();
    return;
  }
  runtime_activity->log(*logger);
}

void XPUActivityProfilerSession::handleKernelActivity(
    const pti_view_record_kernel* activity,
    logger_t* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const itrace_t* linked =
      linkedActivity(activity->_correlation_id, cpuCorrelationMap_);
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      act_t::CONCURRENT_KERNEL,
      std::string(activity->_name));
  auto& kernel_activity = traceBuffer_.activities.back();
  kernel_activity->startTime = activity->_start_timestamp;
  kernel_activity->endTime = activity->_end_timestamp;
  kernel_activity->id = activity->_correlation_id;
  kernel_activity->device = c10::xpu::get_device_idx_from_uuid(activity->_device_uuid);
  kernel_activity->resource = getMappedQueueId(activity->_sycl_queue_id);
  kernel_activity->threadId = activity->_thread_id;
  kernel_activity->flow.id = activity->_correlation_id;
  kernel_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  kernel_activity->flow.start = 0;
  kernel_activity->linked = linked;
  kernel_activity->addMetadata("appended", activity->_append_timestamp);
  kernel_activity->addMetadata("submitted", activity->_submit_timestamp);
  kernel_activity->addMetadata("device", kernel_activity->deviceId());
  kernel_activity->addMetadataQuoted("context", handleToHexString(activity->_context_handle));
  kernel_activity->addMetadata("sycl queue", activity->_sycl_queue_id);
  kernel_activity->addMetadataQuoted("l0 queue", handleToHexString(activity->_queue_handle));
  kernel_activity->addMetadata("correlation", activity->_correlation_id);
  kernel_activity->addMetadata("kernel_id", activity->_kernel_id);
  
  checkTimestampOrder(&*kernel_activity);
  if (outOfRange(*kernel_activity)) {
    traceBuffer_.activities.pop_back();
    return;
  }
  kernel_activity->log(*logger);
}

inline std::string memcpyName(
        pti_view_memcpy_type kind,
        pti_view_memory_type src,
	pti_view_memory_type dst) {
    return fmt::format(
        "Memcpy {} ({} -> {})",
	ptiViewMemcpyTypeToString(kind),
	ptiViewMemoryTypeToString(src),
	ptiViewMemoryTypeToString(dst));
}

template<class pti_view_memory_record_type>
inline std::string bandwidth(pti_view_memory_record_type* activity) {
    auto duration = activity->_end_timestamp - activity->_start_timestamp;
    auto bytes = activity->_bytes;
    return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

void XPUActivityProfilerSession::handleMemcpyActivity(
    const pti_view_record_memory_copy* activity,
    logger_t* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const itrace_t* linked =
      linkedActivity(activity->_correlation_id, cpuCorrelationMap_);
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      act_t::GPU_MEMCPY,
      memcpyName(activity->_memcpy_type, activity->_mem_src, activity->_mem_dst));
  auto& memcpy_activity = traceBuffer_.activities.back();
  memcpy_activity->startTime = activity->_start_timestamp;
  memcpy_activity->endTime = activity->_end_timestamp;
  memcpy_activity->id = activity->_correlation_id;
  memcpy_activity->device = c10::xpu::get_device_idx_from_uuid(activity->_device_uuid);
  memcpy_activity->resource = getMappedQueueId(activity->_sycl_queue_id);
  memcpy_activity->threadId = activity->_thread_id;
  memcpy_activity->flow.id = activity->_correlation_id;
  memcpy_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  memcpy_activity->flow.start = 0;
  memcpy_activity->linked = linked;
  memcpy_activity->addMetadataQuoted("l0 call", std::string(activity->_name));
  memcpy_activity->addMetadata("appended", activity->_append_timestamp);
  memcpy_activity->addMetadata("submitted", activity->_submit_timestamp);
  memcpy_activity->addMetadata("device", memcpy_activity->deviceId());
  memcpy_activity->addMetadataQuoted("context", handleToHexString(activity->_context_handle));
  memcpy_activity->addMetadata("sycl queue", activity->_sycl_queue_id);
  memcpy_activity->addMetadataQuoted("l0 queue", handleToHexString(activity->_queue_handle));
  memcpy_activity->addMetadata("correlation", activity->_correlation_id);
  memcpy_activity->addMetadata("memory opration id", activity->_mem_op_id);
  memcpy_activity->addMetadata("bytes", activity->_bytes);
  memcpy_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));
  
  checkTimestampOrder(&*memcpy_activity);
  if (outOfRange(*memcpy_activity)) {
    traceBuffer_.activities.pop_back();
    return;
  }
  memcpy_activity->log(*logger);
}

void XPUActivityProfilerSession::handleMemsetActivity(
    const pti_view_record_memory_fill* activity,
    logger_t* logger) {
  traceBuffer_.span.opCount += 1;
  traceBuffer_.gpuOpCount += 1;
  const itrace_t* linked =
      linkedActivity(activity->_correlation_id, cpuCorrelationMap_);
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      act_t::GPU_MEMSET,
      fmt::format("Memset ({})", ptiViewMemoryTypeToString(activity->_mem_type)));
  auto& memset_activity = traceBuffer_.activities.back();
  memset_activity->startTime = activity->_start_timestamp;
  memset_activity->endTime = activity->_end_timestamp;
  memset_activity->id = activity->_correlation_id;
  memset_activity->device = c10::xpu::get_device_idx_from_uuid(activity->_device_uuid);
  memset_activity->resource = getMappedQueueId(activity->_sycl_queue_id);
  memset_activity->threadId = activity->_thread_id;
  memset_activity->flow.id = activity->_correlation_id;
  memset_activity->flow.type = libkineto::kLinkAsyncCpuGpu;
  memset_activity->flow.start = 0;
  memset_activity->linked = linked;
  memset_activity->addMetadataQuoted("l0 call", std::string(activity->_name));
  memset_activity->addMetadata("appended", activity->_append_timestamp);
  memset_activity->addMetadata("submitted", activity->_submit_timestamp);
  memset_activity->addMetadata("device", memset_activity->deviceId());
  memset_activity->addMetadataQuoted("context", handleToHexString(activity->_context_handle));
  memset_activity->addMetadata("sycl queue", activity->_sycl_queue_id);
  memset_activity->addMetadataQuoted("l0 queue", handleToHexString(activity->_queue_handle));
  memset_activity->addMetadata("correlation", activity->_correlation_id);
  memset_activity->addMetadata("memory opration id", activity->_mem_op_id);
  memset_activity->addMetadata("bytes", activity->_bytes);
  memset_activity->addMetadata("memory bandwidth (GB/s)", bandwidth(activity));
  
  checkTimestampOrder(&*memset_activity);
  if (outOfRange(*memset_activity)) {
    traceBuffer_.activities.pop_back();
    return;
  }
  memset_activity->log(*logger);
}

void XPUActivityProfilerSession::handleOverheadActivity(
    const pti_view_record_overhead* activity,
    logger_t* logger) {
  traceBuffer_.emplace_activity(
      traceBuffer_.span,
      act_t::OVERHEAD,
      ptiViewOverheadKindToString(activity->_overhead_kind));
  auto& overhead_activity = traceBuffer_.activities.back();
  overhead_activity->startTime = activity->_overhead_start_timestamp_ns;
  overhead_activity->endTime = activity->_overhead_end_timestamp_ns;
  overhead_activity->device = -1;
  overhead_activity->resource = activity->_overhead_thread_id;
  overhead_activity->threadId = activity->_overhead_thread_id;
  overhead_activity->addMetadata("overhead cost", activity->_overhead_duration_ns);
  overhead_activity->addMetadataQuoted("overhead occupancy",
      fmt::format("{}\%", activity->_overhead_duration_ns / overhead_activity->duration()));
  overhead_activity->addMetadata("overhead count", activity->_overhead_count);

  if (outOfRange(*overhead_activity)) {
    return;
  }
  overhead_activity->log(*logger);
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
      handleKernelActivity(
          reinterpret_cast<const pti_view_record_kernel*>(record), logger);
      break;
    case PTI_VIEW_DEVICE_GPU_MEM_COPY:
      handleMemcpyActivity(
          reinterpret_cast<const pti_view_record_memory_copy*>(record), logger);
      break;
    case PTI_VIEW_DEVICE_GPU_MEM_FILL:
      handleMemsetActivity(
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

}
